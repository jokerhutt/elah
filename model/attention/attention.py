import torch
from torch.nn import functional as F

from config import D_MODEL, DROPOUT


class MultiHeadAttention(torch.nn.Module) :

    def __init__(self, num_heads, head_size) :
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.inner_size = num_heads * head_size

        self.qkv = torch.nn.Linear(D_MODEL, 3 * self.inner_size, bias=False)
        self.proj = torch.nn.Linear(self.inner_size, D_MODEL)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x) :
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.inner_size, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,

            # Enable masked attention
            is_causal=True,

            # Only load dropout in SFT
            dropout_p=DROPOUT if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_size)

        return self.dropout(self.proj(out))
