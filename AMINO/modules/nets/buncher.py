import copy
from collections import OrderedDict

import torch.nn as nn

from AMINO.modules.nets.norm import AUDIO_NORM
from AMINO.utils.init_object import init_object

class SIMPLE_LINEAR_BUNCHER(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dims=[256],
        drop_out=0.2,
        norm_layer="batch",
        act={
            "select": "torch.nn:ReLU",
            "conf": {"inplace": True}, 
        },
        last_act={
            "select": "torch.nn:Sigmoid",
            "conf": {}, 
        }
    ):
        super().__init__()
        hidden_dims = copy.deepcopy(hidden_dims)
        hidden_dims.append(num_classes)

        num_layer = len(hidden_dims) - 1
        layers = OrderedDict()
        for i in range(num_layer):
            layers[f'dropout{i}'] = nn.Dropout(p=drop_out, inplace=False)
            layers[f'linear{i}'] = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if i != (num_layer - 1):
                if norm_layer == "batch":
                    layers[f'norm{i}'] = nn.BatchNorm2d(1)
                elif norm_layer == "audio":
                    layers[f'norm{i}'] = AUDIO_NORM(hidden_dims[i+1])
                layers[f'activation{i}'] = init_object(act)
        layers[f'activation{i}'] = init_object(last_act)
        self.net = nn.Sequential(layers)

    def forward(self, xs, xs_len):
        return self.net(xs), xs_len

    def get_num_classes(self):
        return self.net[-1].weight.size(0)
