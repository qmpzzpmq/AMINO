import copy
from collections import OrderedDict

import torch.nn as nn
import torch

from AMINO.utils.init_object import init_object

class SIMPLE_LINEAR_BUNCHER(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dims=[256],
        drop_out=0.2,
        act_fn={
            "select": "torch.nn:Sigmoid",
            "conf": dict(),
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
                layers[f'norm{i}'] = nn.BatchNorm2d(1)
                # layers[f'norm{i}'] = AUDIO_NORM(hidden_dims[num_layer-i-1])
                layers[f'activation{i}'] = nn.ReLU(inplace=True)
        self.net = nn.Sequential(layers)

        #self.net = nn.Sequential(*[
        #    nn.Linear(hidden_dims[i], hidden_dims[i+1]) 
        #    for i in range(len(hidden_dims)-1)
        #])
        self.act = init_object(act_fn)

    def forward(self, xs, xs_len):
        return self.act(self.net(xs)), xs_len

    def get_num_classes(self):
        return self.net[-1].weight.size(0)
