# import sys

import torch
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


class Mlp2(nn.Module):
    def __init__(self, dim_input, e_dim=128):
        super().__init__()
        self.fc = nn.Linear(dim_input, 1000)
        self.embedding = nn.Embedding(13523, e_dim)

    def forward(self, feats):
        x = feats['x']
        c = feats['content']

        c = self.embedding(c)
        c = c.view(len(c), -1)

        x = torch.cat([x, c], dim=1)
        x = self.fc(x)
        return x
