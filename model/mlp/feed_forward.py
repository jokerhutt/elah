import torch

from config import DROPOUT


class FeedForward(torch.nn.Module) :

    def __init__(self, n_embd) :

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)