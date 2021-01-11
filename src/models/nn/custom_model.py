import sys

import layer
import torch.nn as nn
from torch.nn.utils import weight_norm

from . import (mlp, tabnet, transformer, saint, saint_v2, saint_v3, saint_v4, saint_v5,
               saint_v6, saint_v7, saint_v8, saint_v9, saint_v10, saint_v11, saint_v12,
               saint_v13, saint_v14)

sys.path.append('../src')

model_encoder = {
    'mlp': mlp.Mlp,
    'mlp2': mlp.Mlp2,

    'tabnet': tabnet.TabNet,

    'transformer_public': transformer.SAKTModel,
    'transformer_saint': saint.SAINT,
    'transformer_saint_v2': saint_v2.SAINT,
    'transformer_saint_v3': saint_v3.SAINT,
    'transformer_saint_v4': saint_v4.SAINT,
    'transformer_saint_v5': saint_v5.SAINT,
    'transformer_saint_v6': saint_v6.SAINT,
    'transformer_saint_v7': saint_v7.SAINT,
    'transformer_saint_v8': saint_v8.SAINT,
    'transformer_saint_v9': saint_v9.SAINT,
    'transformer_saint_v10': saint_v10.SAINT,
    'transformer_saint_v11': saint_v11.SAINT,
    'transformer_saint_v12': saint_v12.SAINT,
    'transformer_saint_v13': saint_v13.SAINT,
    'transformer_saint_v14': saint_v14.SAINT,
}


def get_head(cfg):
    head_modules = []

    for m in cfg.values():
        if hasattr(nn, m['name']):
            module = getattr(nn, m['name'])(**m['params'])
            if hasattr(m, 'weight_norm'):
                if m['weight_norm']:
                    module = weight_norm(module)
        elif hasattr(layer, m['name']):
            module = getattr(layer, m['name'])(**m['params'])
        head_modules.append(module)

    head_modules = nn.Sequential(*head_modules)

    return head_modules


def replace_fc(model, cfg):
    if cfg.model.backbone.startswith('mlp'):
        model.fc = get_head(cfg.model.head)
    elif cfg.model.backbone.startswith('tabnet'):
        model.final_mapping = get_head(cfg.model.head)
    return model


class CustomModel(nn.Module):
    def __init__(self, cfg):
        super(CustomModel, self).__init__()
        self.cfg = cfg
        self.base_model = model_encoder[cfg.model.backbone](**cfg.model.params)
        self.model = replace_fc(self.base_model, cfg)

    def forward(self, x):
        x = self.model(x)
        return x
