import math
import statistics
import time
from collections import deque
from dataclasses import replace

import numpy as np
import torch

from config import METRICS_DIR, STAGES, TOKEN_DTYPE, ModelConfig, TrainingConfig
from logs import MetricsLog, format_duration, get_logger, log_panel, log_settings, training_progress
from model_io import latest_checkpoint, load_checkpoint, save_checkpoint
from model.attention.attention import MultiHeadAttention
from model.transformer import ElahGPT
from tokenizer.tokenizer import Tokenizer


log = get_logger()


class GradGuard:

    def __init__(self, factor: float, history: int):
        self.factor = factor
        self.norms = deque(maxlen=history)
        self.min_history = max(history // 2, 1)

    def check(self, grad_norm: float):
        if not math.isfinite(grad_norm):
            return False, None

        self.norms.append(grad_norm)

        if len(self.norms) < self.min_history:
            return True, None

        threshold = self.factor * statistics.median(self.norms)

        return grad_norm <= threshold, threshold


class StepWindow:

    def __init__(self):
        self.skipped_total = 0
        self.reset()

    def reset(self):
        self.losses = []
        self.grad_norms = []
        self.skipped = 0
        self.started = time.monotonic()

    def record(self, loss: float, grad_norm: float, skipped: bool):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

        if skipped:
            self.skipped += 1
            self.skipped_total += 1

    def summary(self, tokens_per_step: int):
        if not self.losses:
            return {}

        elapsed = time.monotonic() - self.started
        seconds_per_step = elapsed / len(self.losses)
        finite = [n for n in self.grad_norms if math.isfinite(n)]

        return {
            "running_loss": round(float(np.mean(self.losses)), 6),
            "grad_norm": round(self.grad_norms[-1], 4),
            "grad_norm_mean": round(float(np.mean(finite)), 4) if finite else None,
            "grad_norm_max": round(max(finite), 4) if finite else None,
            "seconds_per_step": round(seconds_per_step, 3),
            "tokens_per_sec": round(tokens_per_step / seconds_per_step) if seconds_per_step else None,
            "skipped": self.skipped,
            "skipped_total": self.skipped_total,
        }


class Trainer:

    def __init__(
        self,
        stage: str = "pretrain",
        tokenizer: Tokenizer | None = None,
        model_config: ModelConfig | None = None,
        training_config: TrainingConfig | None = None,
        resume: bool = False,
        resume_path=None,
    ) :

        self.stage = stage
        self.config = STAGES[stage]
        self.resume = resume or resume_path is not None
        self.resume_path = resume_path
        self.model_config = model_config or ModelConfig(dropout=self.config["dropout"])
        self.train_config = training_config or TrainingConfig()

        self.max_iters = self.config["max_iters"]
        self.accum_steps = self.train_config.batch_size // self.train_config.micro_batch_size

        self.tokenizer = tokenizer or Tokenizer()
        self.stop_token_id = getattr(self.tokenizer, self.config["stop_token"])

        self.data = np.memmap(self.config["tokens"], dtype=TOKEN_DTYPE, mode="r")

        self.train_ranges, self.val_ranges = self._split_ranges(len(self.data))
        self._index = None

        log.info(
            f"stage [bold cyan]{stage}[/] "
            f"tokens=[bold]{len(self.data):,}[/] "
            f"source=[dim]{self.config['tokens'].name}[/] "
            f"steps=[bold]{self.max_iters:,}[/] "
            f"split=[bold]{self._range_tokens(self.train_ranges):,}[/]/"
            f"[bold]{self._range_tokens(self.val_ranges):,}[/]"
        )

    def _split_ranges(self, total: int):
        chunk = self.train_config.val_chunk_tokens
        stride = self.train_config.val_stride

        train, val = [], []

        for index, start in enumerate(range(0, total, chunk)):
            bucket = val if index % stride == stride - 1 else train
            bucket.append((start, min(start + chunk, total)))

        return train, val

    def _range_tokens(self, ranges):
        return sum(end - start for start, end in ranges)

    def _build_index(self, ranges):
        block_size = self.model_config.block_size

        starts = np.array([start for start, _ in ranges], dtype=np.int64)
        spans = np.array([end - start - block_size - 1 for start, end in ranges], dtype=np.int64)

        keep = spans > 0
        starts, spans = starts[keep], spans[keep]

        if not len(starts):
            raise ValueError(f"no window fits block_size={block_size}; corpus or split too small")

        return starts, np.cumsum(spans), int(spans.sum())

    def _ensure_index(self):
        if self._index is None:
            self._index = {
                "train": self._build_index(self.train_ranges),
                "val": self._build_index(self.val_ranges),
            }

    def run_training(self, max_new_tokens: int = 500, max_new_sampling_tokens: int = 200):

        train_config = self.train_config

        model, optimizer_state, start_step = self._build_model()
        self._ensure_index()

        m = model.to(train_config.device)
        forward = torch.compile(m) if train_config.use_compile else m

        learning_rate = self.config["learning_rate"]
        optimizer = torch.optim.AdamW(
            self._parameter_groups(m),
            lr=learning_rate,
            betas=train_config.adam_betas,
            fused=train_config.use_fused_adam,
        )

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        metrics = MetricsLog(METRICS_DIR / f"{self.stage}.jsonl")
        tokens_per_step = train_config.batch_size * self.model_config.block_size
        params = sum(p.numel() for p in m.parameters())

        self._log_run_header(m, params, tokens_per_step, start_step)
        metrics.write(**self._run_record(params, tokens_per_step, start_step))

        window = StepWindow()
        guard = GradGuard(train_config.grad_skip_factor, train_config.grad_skip_history)
        best = {"val_loss": math.inf, "step": None}

        if train_config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with training_progress(train_config.show_progress) as progress:
            task = progress.add_task(
                self.stage,
                total=self.max_iters,
                completed=start_step,
                loss="-",
                grad_norm="-",
                skipped="",
            )

            for step in range(start_step, self.max_iters):

                lr = self._learning_rate_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                if step % train_config.eval_interval == 0 or step == self.max_iters - 1:
                    losses = self.estimate_loss(forward)
                    stats = window.summary(tokens_per_step)

                    if losses["val"] < best["val_loss"]:
                        best = {"val_loss": float(losses["val"]), "step": step}

                    self._log_eval(step, losses, lr, stats)
                    metrics.write(
                        **self._eval_record(step, losses, lr, tokens_per_step),
                        **stats,
                    )

                    progress.update(task, loss=f"{losses['train']:.4f}")
                    window.reset()

                optimizer.zero_grad(set_to_none=True)

                loss_total = torch.zeros((), device=train_config.device)

                for _ in range(self.accum_steps):
                    xb, yb = self._get_batch('train')

                    # forward pass
                    with self._autocast():
                        logits, loss = forward(xb, yb)

                    # backward stays outside autocast
                    (loss / self.accum_steps).backward()

                    loss_total += loss.detach() / self.accum_steps

                grad_norm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), train_config.grad_clip))

                healthy, threshold = guard.check(grad_norm)

                if healthy:
                    optimizer.step()
                else:
                    limit = "non-finite" if threshold is None else f"limit {threshold:.2f}"
                    log.warning(f"skipped step [bold]{step:,}[/] grad_norm=[red]{grad_norm:.2f}[/] [dim]{limit}[/]")

                window.record(float(loss_total), grad_norm, skipped=not healthy)

                progress.update(
                    task,
                    advance=1,
                    grad_norm=f"{grad_norm:.2f}",
                    skipped=f"skipped {window.skipped_total}" if window.skipped_total else "",
                )

                # Sample inference output
                if step % train_config.sample_interval == 0 and step > 0:
                    self._sample_model_output(m, max_new_sampling_tokens, train_config.device, step)

                # Save checkpoint
                if step % train_config.checkpoint_interval == 0 and step > 0:
                    self._save_checkpoint(m, optimizer, step, metrics, best)

        self._save_checkpoint(m, optimizer, self.max_iters, metrics, best)

        metrics.write(
            event="run_end",
            stage=self.stage,
            step=self.max_iters,
            skipped_total=window.skipped_total,
            best_val_loss=None if best["step"] is None else round(best["val_loss"], 6),
            best_step=best["step"],
        )

        log.info(
            f"finished [bold cyan]{self.stage}[/] "
            f"best val [cyan]{best['val_loss']:.4f}[/] at step [bold]{best['step']:,}[/] "
            f"skipped=[bold]{window.skipped_total}[/]"
        )

    def _log_run_header(self, model, params, tokens_per_step, start_step):
        train_config = self.train_config

        log_settings(
            f"{self.stage} run",
            {
                "model": {
                    "params": f"{params/1e6:.1f}M",
                    "layers": self.model_config.n_layer,
                    "heads": self.model_config.n_head,
                    "d_model": self.model_config.d_model,
                    "block_size": self.model_config.block_size,
                    "vocab": f"{self.model_config.vocab_size:,}",
                    "tied_weights": self.model_config.tie_weights,
                    "dropout": self.model_config.dropout,
                },
                "schedule": {
                    "steps": f"{start_step:,} -> {self.max_iters:,}",
                    "peak_lr": f"{self.config['learning_rate']:.1e}",
                    "warmup": f"{train_config.warmup_steps:,} steps",
                    "min_lr": f"{self.config['learning_rate']*train_config.min_lr_ratio:.1e}",
                },
                "batch": {
                    "batch_size": train_config.batch_size,
                    "micro_batch": f"{train_config.micro_batch_size} x {self.accum_steps} accum",
                    "tokens_per_step": f"{tokens_per_step:,}",
                    "token_budget": f"{self.max_iters*tokens_per_step/1e9:.2f}B",
                    "corpus": f"{len(self.data)/1e9:.2f}B",
                    "epochs": f"{self.max_iters*tokens_per_step/len(self.data):.3f}",
                },
                "stability": {
                    "grad_clip": train_config.grad_clip,
                    "skip_above": f"{train_config.grad_skip_factor}x median of {train_config.grad_skip_history}",
                    "weight_decay": train_config.weight_decay,
                    "betas": train_config.adam_betas,
                },
                "runtime": {
                    "device": train_config.device,
                    "bf16": train_config.use_bf16,
                    "compile": train_config.use_compile,
                    "fused_adam": train_config.use_fused_adam,
                },
            },
        )

    def _log_eval(self, step, losses, lr, stats):
        parts = [
            f"step [bold]{step:>7,}[/]",
            f"train [green]{losses['train']:.4f}[/]",
            f"val [cyan]{losses['val']:.4f}[/]",
            f"ppl [cyan]{math.exp(min(float(losses['val']), 20)):,.1f}[/]",
            f"lr [magenta]{lr:.2e}[/]",
        ]

        if stats:
            eta = (self.max_iters - step) * stats["seconds_per_step"]

            mean, peak = stats["grad_norm_mean"], stats["grad_norm_max"]
            gnorm = "nan/nan" if mean is None else f"{mean:.2f}/{peak:.2f}"

            parts += [
                f"gnorm [yellow]{gnorm}[/]",
                f"[dim]{stats['tokens_per_sec'] or 0:,}tok/s[/]",
                f"[dim]eta {format_duration(eta)}[/]",
            ]

            if stats["skipped"]:
                parts.append(f"[red]skipped {stats['skipped']}[/]")

        log.info(" ".join(parts))

    def _run_record(self, params, tokens_per_step, start_step):
        train_config = self.train_config

        return {
            "event": "run_start",
            "stage": self.stage,
            "step": start_step,
            "max_iters": self.max_iters,
            "params": params,
            "model_config": {
                "n_layer": self.model_config.n_layer,
                "n_head": self.model_config.n_head,
                "d_model": self.model_config.d_model,
                "block_size": self.model_config.block_size,
                "vocab_size": self.model_config.vocab_size,
                "dropout": self.model_config.dropout,
                "tie_weights": self.model_config.tie_weights,
                "norm_eps": self.model_config.norm_eps,
            },
            "batch_size": train_config.batch_size,
            "micro_batch_size": train_config.micro_batch_size,
            "accum_steps": self.accum_steps,
            "tokens_per_step": tokens_per_step,
            "token_budget": self.max_iters * tokens_per_step,
            "corpus_tokens": len(self.data),
            "train_tokens": self._range_tokens(self.train_ranges),
            "val_tokens": self._range_tokens(self.val_ranges),
            "val_stride": train_config.val_stride,
            "val_chunk_tokens": train_config.val_chunk_tokens,
            "peak_lr": self.config["learning_rate"],
            "warmup_steps": train_config.warmup_steps,
            "min_lr_ratio": train_config.min_lr_ratio,
            "weight_decay": train_config.weight_decay,
            "adam_betas": list(train_config.adam_betas),
            "grad_clip": train_config.grad_clip,
            "grad_skip_factor": train_config.grad_skip_factor,
            "grad_skip_history": train_config.grad_skip_history,
            "eval_iters": train_config.eval_iters,
            "device": train_config.device,
            "bf16": train_config.use_bf16,
            "compile": train_config.use_compile,
            "fused_adam": train_config.use_fused_adam,
            "resumed": self.resume,
        }

    def _eval_record(self, step, losses, lr, tokens_per_step):
        train_loss = float(losses["train"])
        val_loss = float(losses["val"])

        record = {
            "event": "eval",
            "stage": self.stage,
            "step": step,
            "progress": round(step / self.max_iters, 5),
            "tokens": step * tokens_per_step,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_ppl": round(math.exp(min(train_loss, 20)), 3),
            "val_ppl": round(math.exp(min(val_loss, 20)), 3),
            "val_gap": round(val_loss - train_loss, 6),
            "learning_rate": lr,
        }

        if self.train_config.device == "cuda":
            record["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            record["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

        return record

    def _sample_model_output(self, m: ElahGPT, max_new_sampling_tokens, device, step):
        context = torch.full((1, 1), self.stop_token_id, dtype=torch.long, device=device)

        sample = self.tokenizer.decode(
            m.generate_inference(
                context,
                max_new_tokens=max_new_sampling_tokens,
                eos_token_id=self.stop_token_id,
            )[0].tolist()
        )

        log_panel(sample, f"sample @ step {step:,}")

    def _parameter_groups(self, model: ElahGPT):
        decayed = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        undecayed = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

        return [
            {"params": decayed, "weight_decay": self.train_config.weight_decay},
            {"params": undecayed, "weight_decay": 0.0},
        ]

    def _autocast(self):
        return torch.autocast(
            device_type=self.train_config.device,
            dtype=torch.bfloat16,
            enabled=self.train_config.use_bf16,
        )

    def _learning_rate_at(self, step: int):
        train_config = self.train_config
        peak = self.config["learning_rate"]

        # linear warmup
        if step < train_config.warmup_steps:
            return peak * (step + 1) / train_config.warmup_steps

        # cosine decay from peak down to min_lr_ratio * peak
        progress = (step - train_config.warmup_steps) / max(1, self.max_iters - 1 - train_config.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return peak * (train_config.min_lr_ratio + (1 - train_config.min_lr_ratio) * cosine)

    # returns (model, optimizer state or None, step to start from)
    def _build_model(self):
        if self.resume:
            model, optimizer_state, step = load_checkpoint(
                self.stage,
                self.train_config.device,
                path=self.resume_path,
            )

            self.model_config = model.config
            source = self.resume_path or latest_checkpoint(self.stage).path
            log.info(f"continuing [bold cyan]{self.stage}[/] from [dim]{source}[/] at step [bold]{step:,}[/]")

            return model, optimizer_state, step

        previous = self.config["resume_from"]

        if previous is None:
            return ElahGPT(self.model_config), None, 0

        found = latest_checkpoint(previous)

        if found is None:
            raise FileNotFoundError(f"stage '{self.stage}' resumes from '{previous}', but it has no checkpoints")

        # a new stage starts its own schedule, so only the weights carry over
        model, _, _ = load_checkpoint(previous, self.train_config.device, path=found.path)

        # the earlier stage's architecture wins, so this stage saves what it actually holds
        self.model_config = model.config
        self._apply_dropout(model, self.config["dropout"])
        log.info(f"resumed weights from [dim]{found.path}[/] dropout=[bold]{self.config['dropout']}[/]")

        return model, None, 0

    def _apply_dropout(self, model: ElahGPT, dropout: float):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout
            elif isinstance(module, MultiHeadAttention):
                module.dropout_p = dropout

        self.model_config = replace(self.model_config, dropout=dropout)

    def _save_checkpoint(self, model: ElahGPT, optimizer, step, metrics=None, best=None):
        best = best or {"val_loss": None, "step": None}

        path = save_checkpoint(
            model,
            optimizer,
            step,
            self.stage,
            self.model_config,
            keep=self.train_config.keep_checkpoints,
            protect=() if best["step"] is None else (best["step"],),
            best_val_loss=best["val_loss"],
            best_step=best["step"],
        )

        size = path.stat().st_size / 1e9
        log.info(f"checkpoint [dim]{path}[/] step=[bold]{step:,}[/] [dim]{size:.1f}GB[/]")

        if metrics is not None:
            metrics.write(
                event="checkpoint",
                stage=self.stage,
                step=step,
                path=str(path),
                size_gb=round(size, 3),
                best_step=best["step"],
            )

        return path

    @torch.no_grad
    def estimate_loss(self, model: ElahGPT) :
        out = {}
        model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(self.train_config.eval_iters)
            for k in range(self.train_config.eval_iters) :
                X, Y = self._get_batch(split)
                with self._autocast():
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # get batch_size amount of random contiguous items of length block_size each
    def _get_batch(self, split: str) :
        self._ensure_index()

        starts, cumulative, total = self._index[split]

        block_size = self.model_config.block_size

        draw = np.random.randint(0, total, size=self.train_config.micro_batch_size)
        bucket = np.searchsorted(cumulative, draw, side="right")
        offset = draw - np.where(bucket > 0, cumulative[bucket - 1], 0)

        ix = starts[bucket] + offset

        x = torch.from_numpy(np.stack([self.data[i: i + block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([self.data[i+1: i + block_size + 1] for i in ix]).astype(np.int64))

        x, y = x.to(self.train_config.device), y.to(self.train_config.device)
        return x,y
