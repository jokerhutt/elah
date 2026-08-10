import torch

from config import TRAINING_SPLIT_PERCENTAGE, GPU_DEVICE, LEARNING_RATE, MAX_ITERS, EVAL_INTERVAL, SAMPLE_INTERVAL, \
    BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, CHECKPOINT_INTERVAL
from model.transformer import ElahGPT
from tokenizer.tokenizer import Tokenizer


class Trainer:

    def __init__(self, data: torch.Tensor, vocab_size: int, tokenizer: Tokenizer) :

        n = int(TRAINING_SPLIT_PERCENTAGE*len(data))
        self.training_data = data[:n]
        self.validation_data = data[n:]

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer


    def run_training(self, max_new_tokens: int = 500, max_new_sampling_tokens: int = 200):

        model = ElahGPT()
        print("Model Initialized")

        m = model.to(GPU_DEVICE)
        print(f"Model device set to {GPU_DEVICE}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        print(f"Optimizer set at learning rate of {LEARNING_RATE}")

        for iter in range(MAX_ITERS):

            if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS:
                losses = self.estimate_loss(m)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self._get_batch('train')

            # forward pass
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Sample inference output
            if iter % SAMPLE_INTERVAL == 0 and iter > 0:
                self._sample_model_output(m, max_new_sampling_tokens, GPU_DEVICE)

            # Save checkpoint
            if iter % CHECKPOINT_INTERVAL == 0 and iter > 0:
                self._save_checkpoint()

        ## TODO SAVE MODEL


    def _sample_model_output(self, m: ElahGPT, max_new_sampling_tokens, device):
        context = self.training_data[:1].unsqueeze(0).to(device)
        sample = self.tokenizer.decode(
            m.generate_inference(
                context,
                max_new_tokens=max_new_sampling_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )[0].tolist()
        )

        print("-----------")
        print(f"SAMPLE @ STEP {iter}")
        print("-----------")
        print(sample)

    def _save_checkpoint(self):
        print("TODO")

    @torch.no_grad
    def estimate_loss(self, model: ElahGPT) :
        out = {}
        model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS) :
                X, Y = self._get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # get batch_size amount of random contiguous items of length block_size each
    def _get_batch(self, split: str) :
        data = self.training_data if split == "train" else self.validation_data

        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1: i + BLOCK_SIZE + 1] for i in ix])

        x, y = x.to(GPU_DEVICE), y.to(GPU_DEVICE)
        return x,y