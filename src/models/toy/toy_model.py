import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids):
        return self.layer(input_ids)
