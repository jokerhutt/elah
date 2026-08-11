import math

import numpy as np
import torch

from config import CHECKPOINT_DIR, STAGES, TOKEN_DTYPE, ModelConfig, TrainingConfig
from model_io import load_checkpoint, save_checkpoint
from model.transformer import ElahGPT
from tokenizer.tokenizer import Tokenizer


class Trainer:

    def __init__(
        self,
        stage: str = "pretrain",
        tokenizer: Tokenizer | None = None,
        model_config: ModelConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) :

        self.stage = stage
        self.config = STAGES[stage]
        self.model_config = model_config or ModelConfig()
        self.train_config = training_config or TrainingConfig()

        self.tokenizer = tokenizer or Tokenizer()
        self.stop_token_id = getattr(self.tokenizer, self.config["stop_token"])

        data = np.memmap(self.config["tokens"], dtype=TOKEN_DTYPE, mode="r")

        n = int(self.train_config.training_split*len(data))
        self.training_data = data[:n]
        self.validation_data = data[n:]

        print(f"Stage '{stage}': {len(data):,} tokens from {self.config['tokens'].name}")


    def run_training(self, max_new_tokens: int = 500, max_new_sampling_tokens: int = 200):

        train_config = self.train_config

        model = self._build_model()
        print("Model Initialized")

        m = model.to(train_config.device)
        print(f"Model device set to {train_config.device}")

        learning_rate = self.config["learning_rate"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=train_config.adam_betas,
            weight_decay=train_config.weight_decay,
        )
        print(f"Optimizer set at peak learning rate of {learning_rate}")
        print(f"bfloat16 autocast: {'on' if train_config.use_bf16 else 'off'}")

        for iter in range(train_config.max_iters):

            lr = self._learning_rate_at(iter)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if iter % train_config.eval_interval == 0 or iter == train_config.max_iters - 1:
                losses = self.estimate_loss(m)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")

            xb, yb = self._get_batch('train')

            # forward pass
            with self._autocast():
                logits, loss = m(xb, yb)

            # backward stays outside autocast
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(m.parameters(), train_config.grad_clip)

            optimizer.step()

            # Sample inference output
            if iter % train_config.sample_interval == 0 and iter > 0:
                self._sample_model_output(m, max_new_sampling_tokens, train_config.device, iter)

            # Save checkpoint
            if iter % train_config.checkpoint_interval == 0 and iter > 0:
                self._save_checkpoint(m, optimizer, iter)

        self._save_checkpoint(m, optimizer, train_config.max_iters)


    def _sample_model_output(self, m: ElahGPT, max_new_sampling_tokens, device, step):
        context = torch.tensor(self.training_data[:1].astype(np.int64), device=device).unsqueeze(0)
        sample = self.tokenizer.decode(
            m.generate_inference(
                context,
                max_new_tokens=max_new_sampling_tokens,
                eos_token_id=self.stop_token_id,
            )[0].tolist()
        )

        print("-----------")
        print(f"SAMPLE @ STEP {step}")
        print("-----------")
        print(sample)

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
        progress = (step - train_config.warmup_steps) / max(1, train_config.max_iters - 1 - train_config.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return peak * (train_config.min_lr_ratio + (1 - train_config.min_lr_ratio) * cosine)

    def _checkpoint_path(self, stage: str):
        return CHECKPOINT_DIR / f"{stage}.pt"

    def _build_model(self):
        previous = self.config["resume_from"]

        if previous is None:
            return ElahGPT(self.model_config)

        path = self._checkpoint_path(previous)

        if not path.exists():
            raise FileNotFoundError(f"stage '{self.stage}' resumes from '{previous}', but {path} is missing")

        model, _, _ = load_checkpoint(previous, self.train_config.device)

        # the earlier stage's architecture wins, so this stage saves what it actually holds
        self.model_config = model.config
        print(f"Resumed weights from {path}")

        return model

    def _save_checkpoint(self, model: ElahGPT, optimizer, step):
        path = save_checkpoint(model, optimizer, step, self.stage, self.model_config)
        print(f"Saved checkpoint to {path} at step {step}")

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

        block_size = self.train_config.block_size

        ix = np.random.randint(0, len(data) - block_size, size=self.train_config.batch_size)

        x = torch.from_numpy(np.stack([data[i: i + block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i+1: i + block_size + 1] for i in ix]).astype(np.int64))

        x, y = x.to(self.train_config.device), y.to(self.train_config.device)
        return x,y