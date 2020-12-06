import numpy as np
import torch
from torch.utils.data import Dataset


def numpy_onehot(array, dim_num):
    output = np.identity(dim_num)[array]
    return output


class CustomDataset(Dataset):
    def __init__(self, current, history, target, cfg):
        self.cfg = cfg

        self.history = history
        self.current = current
        self.target = target.reshape(-1, 1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        current_feats = torch.FloatTensor(self.current[idx])
        history_feats = torch.FloatTensor(self.history[idx])

        feats = {
            'current': current_feats,
            'history': history_feats
        }

        if self.target is not None:
            target = torch.FloatTensor(self.target[idx])

        return feats, target
