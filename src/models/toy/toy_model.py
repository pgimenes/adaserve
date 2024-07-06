import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.layer(x)
