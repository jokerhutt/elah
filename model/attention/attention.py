import torch
from torch.nn import functional as F

from config import ModelConfig


class MultiHeadAttention(torch.nn.Module) :

    def __init__(self, config: ModelConfig) :
        super().__init__()

        self.num_heads = config.n_head
        self.head_size = config.d_model // config.n_head
        self.inner_size = self.num_heads * self.head_size
        self.dropout_p = config.dropout

        self.qkv = torch.nn.Linear(config.d_model, 3 * self.inner_size, bias=False)
        self.proj = torch.nn.Linear(self.inner_size, config.d_model)
        self.dropout = torch.nn.Dropout(config.dropout)

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
            dropout_p= self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_size)

        return self.dropout(self.proj(out))
