import torch
import torch.nn as nn

class simple_autoencoder(nn.Module):
    def __init__(
        self,
        feature_dim=257,
        enc_embed_dim=256,
        enc_num_layers=5,
        enc_drop_out=0.3,
        dec_embed_dim=256,
        dec_num_layers=5,
        dec_drop_out=0.3,
    ):
        super().__init__()
        enc_dim = [feature_dim] + [enc_embed_dim] * enc_num_layers
        dec_dim = [dec_embed_dim] * dec_num_layers + [feature_dim]

        enc_layers = []
        for i in range(len(enc_dim)-1):
            enc_layers.append(nn.Dropout(p=enc_drop_out, inplace=False))
            enc_layers.append(nn.Linear(enc_dim[i], enc_dim[i+1]))
            enc_layers.append(nn.ReLU(inplace=True))
        dec_layers = []
        for i in range(len(dec_dim)-1):
            dec_layers.append(nn.Dropout(p=dec_drop_out, inplace=False))
            dec_layers.append(nn.Linear(dec_dim[i], dec_dim[i+1]))
            dec_layers.append(nn.ReLU(inplace=True))
        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y