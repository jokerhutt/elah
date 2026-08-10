import torch

from model.attention.head import Head
from config import D_MODEL, DROPOUT


class MultiHeadAttention(torch.nn.Module) :

    def __init__(self, num_heads, head_size) :
        super().__init__()

        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size * num_heads, D_MODEL)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x) :
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
