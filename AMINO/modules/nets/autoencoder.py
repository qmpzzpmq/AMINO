from collections import OrderedDict
import os
import logging

import h5py

import torch.nn as nn

from AMINO.utils.dynamic_import import dynamic_import, path_convert
from AMINO.utils.hdf5_load import bn2d_load, conv2d_load

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            conv_class="torch.nn:Conv2d",
            pool_size=(2, 2),
            pool_class="torch.nn:AvgPool2d",    
        ):
        super().__init__()
        conv_class = dynamic_import(conv_class)
        pool_class = dynamic_import(pool_class)
        self.net = nn.Sequential(
            conv_class(
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size=(3, 3), stride=(1, 1),
                    padding=(1, 1), bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            conv_class(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=(3, 3), stride=(1, 1),
                padding=(1, 1), bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            pool_class(pool_size),
        )
        self.init_weight()
        
    def init_weight(self):
        for layer in self.net:
            if type(layer) == nn.Conv2d:
                init_layer(layer)
                continue
            elif type(layer) == nn.BatchNorm2d:
                init_bn(layer)
                continue

    def forward(self, x):
        return self.net(x)

class simple_autoencoder(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        hidden_dims= [128, 128, 128, 128, 8],
        enc_drop_out=0.2,
        dec_drop_out=0.2,
        load_from_h5=None
    ):
        super().__init__()
        hidden_dims.insert(0, feature_dim)
        num_layer = len(hidden_dims) - 1
        layers = OrderedDict()
        for i in range(num_layer):
            layers[f'dropout{i}'] = nn.Dropout(p=enc_drop_out, inplace=False)
            layers[f'linear{i}'] = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            layers[f'batchnorm{i}'] = nn.BatchNorm2d(1)
            layers[f'activation{i}'] = nn.ReLU(inplace=True)
        self.enc = nn.Sequential(layers)
        layers = OrderedDict()
        for i in range(num_layer):
            layers[f'dropout{i}'] = nn.Dropout(p=dec_drop_out, inplace=False)
            layers[f'linear{i}'] = nn.Linear(hidden_dims[num_layer-i], hidden_dims[num_layer-i-1])
            if i != (num_layer - 1):
                layers[f'batchnorm{i}'] = nn.BatchNorm2d(1)
                layers[f'activation{i}'] = nn.ReLU(inplace=True)
        self.dec = nn.Sequential(layers)
        if load_from_h5 is not None:
            logging.warning(f"load from h5 not availeble")
            # self.load_from_h5(load_from_h5)
        # self.weight_init()

    def weight_init(self):
        for layers in [self.enc, self.dec]:
            for layer in layers:
                if type(layer) == nn.Linear:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.normal_(layer.bias)
    
    def load_from_h5(self, path):
        path = path_convert(path)
        with h5py.File(path) as h5_file:
            net_weight = h5_file['model_weights']
            for key in net_weight.keys():
                if key.startswith('batch_normalization_'):
                    idx = int(key.split("_", 2)[2])
                    if idx <= 5:
                        bn2d_load(
                            net_weight[key][key],
                            getattr(self.enc, f"batchnorm{idx-1}"),
                        )
                    else:
                        bn2d_load(
                            net_weight[key][key],
                            getattr(self.dec, f"batchnorm{idx-6}"),
                        )
                elif key.startswith('dense_'):
                    idx = int(key.split("_", 1)[1])
                    if idx <= 5:
                        conv2d_load(
                            net_weight[key][key],
                            getattr(self.enc, f"linear{idx-1}")
                        )
                    else:
                        conv2d_load(
                            net_weight[key][key],
                            getattr(self.dec, f"linear{idx-6}"),
                        )

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y

class conv_autoencoder(nn.Module):
    def __init__(
        self,
        channels=[64, 128, 256, 512],
        enc_dropout_p=0.2,
        dec_dropout_p=0.2,
        resume_from_cnn10=None,
    ):
        super().__init__()
        raise NotImplementedError("not implemented yet")
        layers = []
        channels = [1] + channels
        num_layer = len(channels) - 1
        for i in range(num_layer):
            layers.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
            )
            layers.append(
                nn.Dropout(enc_dropout_p)
            )
        self.enc = nn.Sequential(*layers)
        layers = []
        for i in reversed(range(num_layer)):
            layers.append(
                ConvBlock(
                    in_channels=channels[i + 1],
                    out_channels=channels[i],
                    conv_class="torch.nn:ConvTranspose2d",
                )
            )
            layers.append(
                nn.Dropout(dec_dropout_p)
            )
        self.dec = nn.Sequential(*layers)
        if resume_from_cnn10 is not None:
            self.resume_from_cnn10(resume_from_cnn10)

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y
    
    def resume_from_cnn10(self, path):
        pass
