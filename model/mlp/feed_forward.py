import torch

from config import ModelConfig


class FeedForward(torch.nn.Module) :

    def __init__(self, config: ModelConfig) :

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, 4 * config.d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * config.d_model, config.d_model),
            torch.nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)
