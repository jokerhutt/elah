import torch

from config import D_MODEL, BLOCK_SIZE, DROPOUT


class Head (torch.nn.Module):

    def __init__(self, head_size) :
        super().__init__()

        self.key = torch.nn.Linear(D_MODEL, head_size, bias = False)
        self.query = torch.nn.Linear(D_MODEL, head_size, bias = False)
        self.value = torch.nn.Linear(D_MODEL, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x) :

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out