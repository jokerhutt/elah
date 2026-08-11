import torch

from config import ModelConfig
from model.attention.attention import MultiHeadAttention

from model.mlp.feed_forward import FeedForward


class Block(torch.nn.Module) :

    def __init__(self, config: ModelConfig) :

        super().__init__()

        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.attn_norm = torch.nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn_norm = torch.nn.RMSNorm(config.d_model, eps=config.norm_eps)

    # do the self attention + feed forward
    def forward(self, x):
        x = x + self.sa(self.attn_norm(x))
        x = x + self.ffwd(self.ffn_norm(x))
        return x
