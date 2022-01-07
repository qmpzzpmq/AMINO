import numpy as np

import torch.nn as nn

from AMINO.modules.nets.transformer.embedding import PositionalEncoding
from AMINO.modules.nets.transformer.embedding import RelPositionalEncoding
from AMINO.modules.nets.transformer.embedding import NoPositionalEncoding
from AMINO.modules.nets.subsampling import LinearNoSubsampling
from AMINO.modules.nets.transformer.attention import MultiHeadedAttention
# from AMINO.modules.nets.transformer.attention import RelPositionMultiHeadedAttention
from AMINO.modules.nets.transformer.encoder_layer import TransformerEncoderLayer
from AMINO.modules.nets.transformer.positionwise_feed_forward import PositionwiseFeedForward
from AMINO.utils.mask import make_pad_mask

class AMINO_TRANSFORMER_ENCODER(nn.Module):
    def __init__(
        self,
        feature_dim=80,
        d_model=256,
        num_block=12,
        num_layers_reuse=1,
        layer_drop=0.0,
        num_class=None,
        global_cmvn=None,
        dim_feedforward = 2048,
        nhead=8,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        activation='relu',
        layer_norm_eps=1e-05,
        pos_enc_layer_type: str = "abs_pos",
        normalize_before=True,
        concat_after=False,
    ):
        super().__init__()
        self.global_cmvn = None
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        self.embd = LinearNoSubsampling(
            feature_dim, 
            d_model, 
            dropout_rate,
            pos_enc_class(d_model, positional_dropout_rate),
        )
        # self.encoders = nn.ModuleList([
        #     nn.TransformerEncoderLayer(
        #             d_model=d_model,
        #             nhead=nhead,
        #             dim_feedforward=dim_feedforward, 
        #             dropout=dropout,
        #             activation=activation,
        #             layer_norm_eps=layer_norm_eps,
        #     ) for _ in range(num_block)
        # ])
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                MultiHeadedAttention(
                    nhead, 
                    d_model,
                    attention_dropout_rate,
                ),
                PositionwiseFeedForward(
                    d_model, 
                    dim_feedforward,
                    attention_dropout_rate,
                ),
                attention_dropout_rate,
                normalize_before,
                concat_after,
                layer_norm_eps=layer_norm_eps,
            ) for _ in range(num_block)
        ])
        self.num_layers_reuse = num_layers_reuse
        if layer_drop > 0.0 and num_layers_reuse > 1:
            assert layer_drop < 1.0, f"""
                layer_drop should [0.0, 1.0),
                now it is {layer_drop}
            """
            self.layer_drop = layer_drop

    def forward(self, xs, xs_lens):
        num_layer_reuse = np.random.binomial(
            self.num_layer_reuse, self.layer_drop
        ) \
            if self.training and hasattr(self, "layer_drop") \
            else self.num_layer_reuse
        masks = ~make_pad_mask(xs_lens)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embd(xs, masks)
        for _ in range(num_layer_reuse):
            for encoder in self.encoders:
                xs = encoder(xs, masks)
        return xs, masks
