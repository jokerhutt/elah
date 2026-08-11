import torch

from config import NORM_EPS
from model.attention.attention import MultiHeadAttention

from model.mlp.feed_forward import FeedForward


class Block(torch.nn.Module) :

    def __init__(self, n_embd, n_head) :

        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.attn_norm = torch.nn.RMSNorm(n_embd, eps=NORM_EPS)
        self.ffn_norm = torch.nn.RMSNorm(n_embd, eps=NORM_EPS)

    # do the self attention + feed forward
    def forward(self, x):
        x = x + self.sa(self.attn_norm(x))
        x = x + self.ffwd(self.ffn_norm(x))
        return x