import math

import numpy as np
import torch

from config import TRAINING_SPLIT_PERCENTAGE, GPU_DEVICE, MAX_ITERS, EVAL_INTERVAL, SAMPLE_INTERVAL, \
    BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, CHECKPOINT_INTERVAL, CHECKPOINT_DIR, STAGES, TOKEN_DTYPE, \
    USE_BF16, WARMUP_STEPS, MIN_LR_RATIO, GRAD_CLIP, ADAM_BETAS, WEIGHT_DECAY
from model.transformer import ElahGPT
from tokenizer.tokenizer import Tokenizer


class Trainer:

    def __init__(self, stage: str = "pretrain", tokenizer: Tokenizer | None = None) :

        self.stage = stage
        self.config = STAGES[stage]

        self.tokenizer = tokenizer or Tokenizer()
        self.stop_token_id = getattr(self.tokenizer, self.config["stop_token"])

        data = np.memmap(self.config["tokens"], dtype=TOKEN_DTYPE, mode="r")

        n = int(TRAINING_SPLIT_PERCENTAGE*len(data))
        self.training_data = data[:n]
        self.validation_data = data[n:]

        print(f"Stage '{stage}': {len(data):,} tokens from {self.config['tokens'].name}")


    def run_training(self, max_new_tokens: int = 500, max_new_sampling_tokens: int = 200):

        model = ElahGPT()
        print("Model Initialized")

        m = model.to(GPU_DEVICE)
        print(f"Model device set to {GPU_DEVICE}")

        self._resume_from_previous_stage(m)

        learning_rate = self.config["learning_rate"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=ADAM_BETAS,
            weight_decay=WEIGHT_DECAY,
        )
        print(f"Optimizer set at peak learning rate of {learning_rate}")
        print(f"bfloat16 autocast: {'on' if USE_BF16 else 'off'}")

        for iter in range(MAX_ITERS):

            lr = self._learning_rate_at(iter)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
                losses = self.estimate_loss(m)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")

            xb, yb = self._get_batch('train')

            # forward pass
            with self._autocast():
                logits, loss = m(xb, yb)

            # backward stays outside autocast
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)

            optimizer.step()

            # Sample inference output
            if iter % SAMPLE_INTERVAL == 0 and iter > 0:
                self._sample_model_output(m, max_new_sampling_tokens, GPU_DEVICE, iter)

            # Save checkpoint
            if iter % CHECKPOINT_INTERVAL == 0 and iter > 0:
                self._save_checkpoint(m, optimizer, iter)

        self._save_checkpoint(m, optimizer, MAX_ITERS)


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
        return torch.autocast(device_type=GPU_DEVICE, dtype=torch.bfloat16, enabled=USE_BF16)

    def _learning_rate_at(self, step: int):
        peak = self.config["learning_rate"]

        # linear warmup
        if step < WARMUP_STEPS:
            return peak * (step + 1) / WARMUP_STEPS

        # cosine decay from peak down to MIN_LR_RATIO * peak
        progress = (step - WARMUP_STEPS) / max(1, MAX_ITERS - 1 - WARMUP_STEPS)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return peak * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine)

    def _checkpoint_path(self, stage: str):
        return CHECKPOINT_DIR / f"{stage}.pt"

    def _resume_from_previous_stage(self, model: ElahGPT):
        previous = self.config["resume_from"]

        if previous is None:
            return

        path = self._checkpoint_path(previous)

        if not path.exists():
            raise FileNotFoundError(f"stage '{self.stage}' resumes from '{previous}', but {path} is missing")

        model.load_state_dict(torch.load(path, map_location=GPU_DEVICE)["model"])
        print(f"Resumed weights from {path}")

    def _save_checkpoint(self, model: ElahGPT, optimizer, step):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(self.stage)

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            path,
        )
        print(f"Saved checkpoint to {path} at step {step}")

    @torch.no_grad
    def estimate_loss(self, model: ElahGPT) :
        out = {}
        model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS) :
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

        ix = np.random.randint(0, len(data) - BLOCK_SIZE, size=BATCH_SIZE)

        x = torch.from_numpy(np.stack([data[i: i + BLOCK_SIZE] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i+1: i + BLOCK_SIZE + 1] for i in ix]).astype(np.int64))

        x, y = x.to(GPU_DEVICE), y.to(GPU_DEVICE)
        return x,y