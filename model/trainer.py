import math
from dataclasses import replace

import numpy as np
import torch

from config import CHECKPOINT_DIR, STAGES, TOKEN_DTYPE, ModelConfig, TrainingConfig
from logs import get_logger, log_panel, training_progress
from model_io import load_checkpoint, save_checkpoint
from model.attention.attention import MultiHeadAttention
from model.transformer import ElahGPT
from tokenizer.tokenizer import Tokenizer


log = get_logger()


class Trainer:

    def __init__(
        self,
        stage: str = "pretrain",
        tokenizer: Tokenizer | None = None,
        model_config: ModelConfig | None = None,
        training_config: TrainingConfig | None = None,
        resume: bool = False,
    ) :

        self.stage = stage
        self.config = STAGES[stage]
        self.resume = resume
        self.model_config = model_config or ModelConfig(dropout=self.config["dropout"])
        self.train_config = training_config or TrainingConfig()

        self.max_iters = self.config["max_iters"]

        self.tokenizer = tokenizer or Tokenizer()
        self.stop_token_id = getattr(self.tokenizer, self.config["stop_token"])

        data = np.memmap(self.config["tokens"], dtype=TOKEN_DTYPE, mode="r")

        n = int(self.train_config.training_split*len(data))
        self.training_data = data[:n]
        self.validation_data = data[n:]

        log.info(
            f"stage [bold cyan]{stage}[/] "
            f"tokens=[bold]{len(data):,}[/] "
            f"source=[dim]{self.config['tokens'].name}[/] "
            f"steps=[bold]{self.max_iters:,}[/]"
        )


    def run_training(self, max_new_tokens: int = 500, max_new_sampling_tokens: int = 200):

        train_config = self.train_config

        model, optimizer_state, start_step = self._build_model()

        m = model.to(train_config.device)

        learning_rate = self.config["learning_rate"]
        optimizer = torch.optim.AdamW(
            self._parameter_groups(m),
            lr=learning_rate,
            betas=train_config.adam_betas,
        )

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        log.info(
            f"params=[bold]{sum(p.numel() for p in m.parameters())/1e6:.1f}M[/] "
            f"device=[bold]{train_config.device}[/] "
            f"lr=[magenta]{learning_rate:.1e}[/] "
            f"bf16=[bold]{'on' if train_config.use_bf16 else 'off'}[/]"
        )

        with training_progress(train_config.show_progress) as progress:
            task = progress.add_task(self.stage, total=self.max_iters, completed=start_step, loss="-")

            for step in range(start_step, self.max_iters):

                lr = self._learning_rate_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                if step % train_config.eval_interval == 0 or step == self.max_iters - 1:
                    losses = self.estimate_loss(m)
                    log.info(
                        f"step [bold]{step:>7,}[/] "
                        f"train [green]{losses['train']:.4f}[/] "
                        f"val [cyan]{losses['val']:.4f}[/] "
                        f"lr [magenta]{lr:.2e}[/]"
                    )
                    progress.update(task, loss=f"{losses['train']:.4f}")

                xb, yb = self._get_batch('train')

                # forward pass
                with self._autocast():
                    logits, loss = m(xb, yb)

                # backward stays outside autocast
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(m.parameters(), train_config.grad_clip)

                optimizer.step()

                progress.update(task, advance=1)

                # Sample inference output
                if step % train_config.sample_interval == 0 and step > 0:
                    self._sample_model_output(m, max_new_sampling_tokens, train_config.device, step)

                # Save checkpoint
                if step % train_config.checkpoint_interval == 0 and step > 0:
                    self._save_checkpoint(m, optimizer, step)

        self._save_checkpoint(m, optimizer, self.max_iters)


    def _sample_model_output(self, m: ElahGPT, max_new_sampling_tokens, device, step):
        context = torch.tensor(self.training_data[:1].astype(np.int64), device=device).unsqueeze(0)
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

    def _checkpoint_path(self, stage: str):
        return CHECKPOINT_DIR / f"{stage}.pt"

    # returns (model, optimizer state or None, step to start from)
    def _build_model(self):
        if self.resume:
            path = self._checkpoint_path(self.stage)

            if not path.exists():
                raise FileNotFoundError(f"cannot continue stage '{self.stage}', {path} is missing")

            model, optimizer_state, step = load_checkpoint(self.stage, self.train_config.device)
            self.model_config = model.config
            log.info(f"continuing [bold cyan]{self.stage}[/] from [dim]{path}[/] at step [bold]{step:,}[/]")

            return model, optimizer_state, step

        previous = self.config["resume_from"]

        if previous is None:
            return ElahGPT(self.model_config), None, 0

        path = self._checkpoint_path(previous)

        if not path.exists():
            raise FileNotFoundError(f"stage '{self.stage}' resumes from '{previous}', but {path} is missing")

        # a new stage starts its own schedule, so only the weights carry over
        model, _, _ = load_checkpoint(previous, self.train_config.device)

        # the earlier stage's architecture wins, so this stage saves what it actually holds
        self.model_config = model.config
        self._apply_dropout(model, self.config["dropout"])
        log.info(f"resumed weights from [dim]{path}[/] dropout=[bold]{self.config['dropout']}[/]")

        return model, None, 0

    def _apply_dropout(self, model: ElahGPT, dropout: float):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout
            elif isinstance(module, MultiHeadAttention):
                module.dropout_p = dropout

        self.model_config = replace(self.model_config, dropout=dropout)

    def _save_checkpoint(self, model: ElahGPT, optimizer, step):
        path = save_checkpoint(model, optimizer, step, self.stage, self.model_config)
        log.info(f"checkpoint [dim]{path}[/] step=[bold]{step:,}[/]")

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
        data = self.training_data if split == "train" else self.validation_data

        block_size = self.model_config.block_size

        ix = np.random.randint(0, len(data) - block_size, size=self.train_config.batch_size)

        x = torch.from_numpy(np.stack([data[i: i + block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i+1: i + block_size + 1] for i in ix]).astype(np.int64))

        x, y = x.to(self.train_config.device), y.to(self.train_config.device)
        return x,y