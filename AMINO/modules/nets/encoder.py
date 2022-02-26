import logging
import copy

import numpy as np
import hydra

import torch.nn as nn
import transformers

from AMINO.modules.nets.transformer.embedding import PositionalEncoding
from AMINO.modules.nets.transformer.embedding import RelPositionalEncoding
from AMINO.modules.nets.transformer.embedding import NoPositionalEncoding
from AMINO.modules.nets.subsampling import LinearNoSubsampling
from AMINO.modules.nets.transformer.attention import MultiHeadedAttention
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
        num_layers_reuse = np.random.binomial(
            self.num_layers_reuse, self.layer_drop
        ) \
            if self.training and hasattr(self, "layer_drop") \
            else self.num_layers_reuse
        masks = ~make_pad_mask(xs_lens).unsqueeze(1) 
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, xs_len, pos_emb, masks = self.embd(xs, xs_lens, masks)
        for _ in range(num_layers_reuse):
            for encoder in self.encoders:
                xs, masks, _ = encoder(xs, masks, pos_emb)
        return xs, xs_len


class HUGGINGFACE_WAV2VEC2(nn.Module):
    def __init__(
        self,
        config={
            "vocab_size": 32,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": 'gelu',
            "hidden_dropout": 0.1,
            "activation_dropout": 0.1,
            "attention_dropout": 0.1,
            "feat_proj_dropout": 0.0,
            "feat_quantizer_dropout": 0.0,
            "final_dropout": 0.1,
            "layerdrop": 0.1,
            "initializer_range ": 0.02,
            "layer_norm_eps ": 1e-05,
            "feat_extract_norm ": 'group',
            "feat_extract_activation ": 'gelu',
            "conv_dim ": (512, 512, 512, 512, 512, 512, 512),
            "conv_stride ": (5, 2, 2, 2, 2, 2, 2),
            "conv_kernel ": (10, 3, 3, 3, 3, 2, 2),
            "conv_bias ": False,
            "num_conv_pos_embeddings ": 128,
            "num_conv_pos_embedding_groups ": 16,
            "do_stable_layer_norm ": False,
            "apply_spec_augment ": True,
            "mask_time_prob ": 0.05,
            "mask_time_length ": 10,
            "mask_time_min_masks ": 2,
            "mask_feature_prob ": 0.0,
            "mask_feature_length ": 10,
            "mask_feature_min_masks ": 0,
            "num_codevectors_per_group ": 320,
            "num_codevector_groups ": 2,
            "contrastive_logits_temperature ": 0.1,
            "num_negatives ": 100,
            "codevector_dim ": 256,
            "proj_codevector_dim ": 256,
            "diversity_loss_weight ": 0.1,
            "ctc_loss_reduction ": 'sum',
            "ctc_zero_infinity ": False,
            "use_weighted_layer_sum ": False,
            "classifier_proj_size ": 256,
            "tdnn_dim ": (512, 512, 512, 512, 1500),
            "tdnn_kernel ": (5, 3, 3, 1, 1),
            "tdnn_dilation ": (1, 2, 3, 1, 1),
            "xvector_output_dim ": 512,
            "pad_token_id ": 0,
            "bos_token_id ": 1,
            "eos_token_id ": 2,
            "add_adapter ": False,
            "adapter_kernel_size": 3,
            "adapter_stride": 2,
            "num_adapter_layers": 3,
            "output_hidden_size": None,
        },
        from_pretrained="facebook/wav2vec2-base", # "facebook/wav2vec2-base-960h",
        from_pretrained_num_hidden_layers=3,
    ):
        assert (config is not None) or (from_pretrained is not None), \
            f"config and from pretrained both is none"
        super().__init__()
        if from_pretrained is not None:
            cache_dir = hydra.utils.to_absolute_path("../../.HF_CACHE")
            logging.info(
                f"using dir {cache_dir} as Hugging Face cache dir"
            )
            pretrain_model = transformers.Wav2Vec2Model.from_pretrained(
                from_pretrained, cache_dir=cache_dir,
            )
        if pretrain_model is not None and config is None:
            pconfig = pretrain_model.config.to_dict()
            num_hidden_layers = pconfig["num_hidden_layers"]
            if from_pretrained_num_hidden_layers is not None:
                assert from_pretrained_num_hidden_layers <= num_hidden_layers, \
                    f"from_pretrained_num_hidden_layers should equal or \
                    samll than num_hidden_layers {num_hidden_layers} in config"
                pretrain_model.encoder.layers = \
                    pretrain_model.encoder.layers[:from_pretrained_num_hidden_layers]
                pretrain_model.config.num_hidden_layers = from_pretrained_num_hidden_layers
            # if hasattr(pretrain_model, "masked_spec_embed"):
            #     logging.info("del masked_spec_embed in pretrain model")
            #     del pretrain_model.masked_spec_embed
            model = pretrain_model
        elif pretrain_model is not None and config is not None:
            raise NotImplementedError("config is not implement yet")
            if config == pretrain_model.config:
                model = pretrain_model
            else:
                temp_pconfig = copy.deepcopy(pretrain_model.config.to_dict())
                temp_pconfig.pop("num_hidden_layers")
                temp_config = copy.deepcopy(config)
                num_hidden_layers = temp_config.pop("num_hidden_layers")
                assert temp_pconfig == temp_config, \
                    "pretrain config should be same except 'num_hidden_layers'"
                pretrain_model.encoder.layers = pretrain_model.encoder.layers[:num_hidden_layers]
        model.train()
        self.net = model
    
    def forward(self, xs, xs_lens):
        # masks = ~make_pad_mask(xs_lens).unsqueeze(1)
        masks = ~make_pad_mask(xs_lens)
        output = self.net(
            input_values = xs,
            attention_mask = masks,
        )
        return output.last_hidden_state, xs_lens
