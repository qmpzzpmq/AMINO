from collections import OrderedDict
import logging

import h5py
import hydra

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from AMINO.utils.hdf5_load import bn2d_load, conv2d_load
from AMINO.modules.nets.cmvn import GlobalCMVN
from AMINO.modules.nets.base_net import AMINO_BASE_NET


class AUDIO_NORM(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features=num_features)

    def forward(self, x):
        return nn.BatchNorm2d.forward(
            self, x.transpose(1, 3),
        ).transpose(1, 3)

class simple_autoencoder(AMINO_BASE_NET):
    def __init__(
        self,
        feature_dim=128,
        hidden_dims= [128, 128, 128, 128, 8],
        enc_drop_out=0.2,
        dec_drop_out=0.2,
        load_from_h5=None,
        sliding_window_cmn=False,
        cmvn_path=None,
    ):
        super().__init__()
        hidden_dims.insert(0, feature_dim)
        num_layer = len(hidden_dims) - 1
        layers = OrderedDict()
        if sliding_window_cmn:
            layers['cmn'] = torchaudio.transforms.SlidingWindowCmn()
        elif cmvn_path:
            cmvn = torch.load(
                hydra.utils.to_absolute_path(cmvn_path),
                map_location=torch.device('cpu')
            )
            assert cmvn['normal']['mean'].size(-1) == feature_dim, \
                f"cmvn feature_dim not equal cmvn readin"
            layers['cmvn'] = GlobalCMVN(
                mean=cmvn['normal']['mean'],
                istd=cmvn['normal']['var'],
            )
        for i in range(num_layer):
            layers[f'dropout{i}'] = nn.Dropout(p=enc_drop_out, inplace=False)
            layers[f'linear{i}'] = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            layers[f'norm{i}'] = nn.BatchNorm2d(1)
            # layers[f'norm{i}'] = AUDIO_NORM(hidden_dims[i+1])
            layers[f'activation{i}'] = nn.ReLU(inplace=True)
        self.enc = nn.Sequential(layers)
        layers = OrderedDict()
        for i in range(num_layer):
            layers[f'dropout{i}'] = nn.Dropout(p=dec_drop_out, inplace=False)
            layers[f'linear{i}'] = nn.Linear(hidden_dims[num_layer-i], hidden_dims[num_layer-i-1])
            if i != (num_layer - 1):
                layers[f'norm{i}'] = nn.BatchNorm2d(1)
                # layers[f'norm{i}'] = AUDIO_NORM(hidden_dims[num_layer-i-1])
                layers[f'activation{i}'] = nn.ReLU(inplace=True)
        self.dec = nn.Sequential(layers)
        if load_from_h5 is not None:
            self.load_from_h5(load_from_h5)
        # self.weight_init()

    def weight_init(self):
        for layers in [self.enc, self.dec]:
            for layer in layers:
                if type(layer) == nn.Linear:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.normal_(layer.bias)
    
    def load_from_h5(self, path):
        path = hydra.utils.to_absolute_path(path)
        with h5py.File(path) as h5_file:
            net_weight = h5_file['model_weights']
            for key in net_weight.keys():
                if key.startswith('batch_normalization_'):
                    idx = int(key.split("_", 2)[2])
                    if idx <= 5:
                        bn2d_load(
                            net_weight[key][key],
                            getattr(self.enc, f"norm{idx-1}"),
                        )
                        logging.warning(f"loading {key} to encoder norm{idx-1}")
                    else:
                        bn2d_load(
                            net_weight[key][key],
                            getattr(self.dec, f"norm{idx-6}"),
                        )
                        logging.warning(f"loading {key} to decoder norm{idx-6}")
                elif key.startswith('dense_'):
                    idx = int(key.split("_", 1)[1])
                    if idx <= 5:
                        conv2d_load(
                            net_weight[key][key],
                            getattr(self.enc, f"linear{idx-1}")
                        )
                        logging.warning(f"loading {key} to encoder linear{idx-1}")
                    else:
                        conv2d_load(
                            net_weight[key][key],
                            getattr(self.dec, f"linear{idx-6}"),
                        )
                        logging.warning(f"loading {key} to decoder linear{idx-6}")

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y


class AMINO_AEGMM(AMINO_BASE_NET):
    def __init__(
        self,
        num_gmm = 4,
        latent_dim=10, # 2 loss+ middle hidden dims
        feature_dim=128,
        hidden_dims= [128, 128, 128, 8], # [128, 128, 128, 128, 8]
        enc_drop_out=0.2,
        dec_drop_out=0.2,
    ):
        """Network for AEGMM"""
        super().__init__()
        hidden_dims.insert(0, feature_dim)
        num_layer = len(hidden_dims) - 1

        # Encoder network
        layers = []
        for i in range(num_layer):
            layers.append(nn.Dropout(p=enc_drop_out, inplace=False))
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            #layers.append(nn.BatchNorm2d(1))
            layers.append(nn.ReLU(inplace=True))
        self.enc = nn.Sequential(*layers)

        # Decoder network
        layers = []
        for i in reversed(range(num_layer)):
            layers.append(nn.Dropout(p=dec_drop_out, inplace=False))
            layers.append(nn.Linear(hidden_dims[i+1], hidden_dims[i]))
            #layers.append(nn.BatchNorm2d(1))
            layers.append(nn.ReLU(inplace=True))
        self.dec = nn.Sequential(*layers)

        # Estimation network
        layers = []
        layers += [nn.Linear(latent_dim, 128)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(128, num_gmm)]
        layers += [nn.Softmax(dim=-1)]
        self.estimation = nn.Sequential(*layers)

    def weight_init(self):
        for layers in [self.enc, self.dec]:
            for layer in layers:
                if type(layer) == nn.Linear:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.normal_(layer.bias)

    def encoder (self, x):
        # x: (B, C, T, F)
        h  = self.enc(x)
        return h

    def decoder(self, h):
        y = self.dec(h)
        return y

    def estimate(self, z):
        gamma = self.estimation(z)
        return gamma

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=-1,) / (
            x.norm(2, dim=-1) + torch.finfo(torch.float32).eps
        )
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=-1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=-1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma

    def is_classifier():
        return False 
