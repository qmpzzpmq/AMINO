import torch.nn as nn

from AMINO.utils.init_object import init_object
from AMINO.modules.nets.buncher import SIMPLE_LINEAR_BUNCHER
from AMINO.utils.mask import make_pad_mask

class AMINO_CLASSIFIER(nn.Module):
    def get_num_classes(self):
        pass

class AMINO_GP_DECODER(AMINO_CLASSIFIER):
    def __init__(
        self,
        buncher,
        pooler,
    ):
        super().__init__()
        self.buncher = init_object(buncher)
        self.pooler = init_object(pooler)
        
    def forward(self, hs, hs_len):
        # hs: (B, T, H)
        cs, cs_len = self.buncher(hs, hs_len)
        # cs: (B, T, C)
        ys, ys_len = self.pooler(cs, cs_len)
        # ys: (B, C)
        return ys, ys_len

    def get_num_classes(self):
        return self.buncher.get_num_classes()

class AMINO_AUTOENCODER_DECODER(nn.Module):
    def foo(self):
        pass

class SIMPLE_LINEAR_AUTOENCODER_DECODER(
        SIMPLE_LINEAR_BUNCHER, AMINO_AUTOENCODER_DECODER
    ):
    def __init__(
        self,
        feature_dim,
        hidden_dims=[256],
        drop_out=0.2,
        act_fn={
            "select": "torch.nn:Sigmoid",
            "conf": dict(),
        }
    ):
        super().__init__(feature_dim, hidden_dims, drop_out, act_fn)

    # def get_num_classes(self):
    #     raise ValidationError("please use AMINO.modules.nets.buncher.SIMPLE_LINEAR_BUNCHER")

# borrow idea from MAE
class TRANSFORMER_ENCODER_AS_AUTTOENDER_DECODER(AMINO_AUTOENCODER_DECODER):
    def __init__(
        self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
    ):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, xs, xs_len):
        masks = ~make_pad_mask(xs_len)
        out = self.net(xs, masks)
        return out, xs_len
