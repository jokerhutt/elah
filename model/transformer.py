from typing import Any

import torch

from config import D_MODEL, VOCAB_SIZE, BLOCK_SIZE, DROPOUT, N_HEAD, N_LAYER, NORM_EPS, GPU_DEVICE
from block import Block


class ElahGPT(torch.nn.Module):

    def __init__(self):

        self.token_embedding_table = torch.nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding_table = torch.nn.Embedding(BLOCK_SIZE, D_MODEL)

        self.blocks = torch.nn.Sequential(*[Block(D_MODEL, N_HEAD) for _ in range (N_LAYER)])

        self.final_norm = torch.nn.RMSNorm(D_MODEL, eps=NORM_EPS)
        self.lm_head = torch.nn.Linear(D_MODEL, VOCAB_SIZE)

        self.apply(self.__init_weights)

    def _init_weights(self, module):
        # Initialize Linear Layer
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Initialise Embedding Layer
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=GPU_DEVICE))

        # Add token and position embedding tensors
        x = tok_emb + pos_emb

        # Forward pass for blocks
        x = self.blocks(x)
        # Forward pass for final norm
        x = self.final_norm(x)
        # Forward pass for lm head
        logits = self.lm_head(x)

        # Loss Computation
        if targets is None:
            loss = None
        else:
            # Reshape logits
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    # === INFERENCE ===
    def generate_inference(
            self,
            idx: torch.Tensor,
            eos_token_id: int | None = None,
            temperature: float = 1.0,
            top_k: int | None = None,
            top_p: int | None = None,
            max_new_tokens: int = 512
    ):
        if temperature <= 0:
            raise ValueError("Temperature must be above zero")

        # save if training and switch to eval mode
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):

                    # trim context to context window
                    idx_cond = idx[:, -BLOCK_SIZE:]

                    # run forward pass
                    logits, _ = self(idx_cond)

                    # get the scores for the next token
                    logits = logits[:, -1, :] / temperature

                    # Trim logits to keep k top next token scores
                    if top_k is not None:
                        k = min(top_k, logits.size(-1))
                        if k <= 0:
                            raise ValueError("top_k must be greater than error")
                        threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))

                    # Trim logits to amount of probable next tokens
                    if top_p is not None:
                        if not 0 < top_p <= 1:
                            raise ValueError("top_p must be in the range (0, 1]")
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        remove = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        ) > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

                    # compute softmax probabilities
                    probs = torch.softmax(logits, dim=-1)

                    # get the next token
                    idx_next = torch.multinomial(probs, num_samples=1)

                    # append the next token to sequence
                    idx = torch.cat((idx, idx_next), dim=1)

                    # if EOS terminate early
                    if eos_token_id is not None and torch.all(idx_next == eos_token_id):
                        break

        finally:
            # if was training, go back to training mode
            self.train(was_training)

        return idx












