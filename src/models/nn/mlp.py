# import sys

# import torch
import torch.nn as nn
# from torch.nn.utils import weight_norm

# sys.path.append('../src')
# import const


class Mlp(nn.Module):
    def __init__(self, dim_input):
        super().__init__()
        self.fc = nn.Linear(dim_input, 1000)

    def forward(self, x):
        x = self.fc(x)
        return x
