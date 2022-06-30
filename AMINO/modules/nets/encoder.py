import logging
import copy
from collections import OrderedDict

import h5py
import numpy as np
import hydra

import torchaudio
import torch
import torch.nn as nn
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

from AMINO.modules.nets.transformer.embedding import PositionalEncoding
from AMINO.modules.nets.transformer.embedding import RelPositionalEncoding
from AMINO.modules.nets.transformer.embedding import NoPositionalEncoding
from AMINO.modules.nets.subsampling import LinearNoSubsampling
from AMINO.modules.nets.transformer.attention import MultiHeadedAttention
from AMINO.modules.nets.transformer.encoder_layer import TransformerEncoderLayer
from AMINO.modules.nets.transformer.positionwise_feed_forward import PositionwiseFeedForward
from AMINO.modules.nets.cmvn import GlobalCMVN
from AMINO.modules.nets.drop_layer import Wav2Vec2EncoderLayer_DropLayer_Warraper
from AMINO.utils.mask import make_pad_mask
from AMINO.utils.hdf5_load import bn2d_load, conv2d_load
from AMINO.utils.init_object import init_object
from AMINO.modules.nets.norm import AUDIO_NORM

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
        drop_layer_p=0.0,
        quantizerVQ_topping=False,
    ):
        assert (config is not None) or (from_pretrained is not None), \
            f"config and from pretrained both is none"
        super().__init__()
        model_class = transformers.Wav2Vec2ForPreTraining \
            if quantizerVQ_topping else transformers.Wav2Vec2Model
        
        if from_pretrained is not None:
            cache_dir = hydra.utils.to_absolute_path("../../.HF_CACHE")
            logging.info(
                f"using dir {cache_dir} as Hugging Face cache dir"
            )
            pretrain_model = model_class.from_pretrained(
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
        elif pretrain_model is None and config is not None:
            model = model_class(config)

        if drop_layer_p > 0.0:
            model.encoder.layers = nn.ModuleList([
                Wav2Vec2EncoderLayer_DropLayer_Warraper(drop_layer_p, x) \
                for x in model.encoder.layers
            ])

        # model.train()
        self.net = model
    
    def forward(self, xs, xs_len, mask_time_indices=None):
        xs = xs.squeeze(1) # delete channel
        masks = ~make_pad_mask(xs_len)
        if mask_time_indices:
            batch_size, raw_sequence_length = xs.shape
            sequence_length = self.net._get_feat_extract_output_lengths(raw_sequence_length)
            mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)
            mask_time_indices = mask_time_indices.to(
                device=xs.device, dtype=torch.long
            )
       
        output = self.net(
            input_values = xs,
            attention_mask = masks,
            mask_time_indices=mask_time_indices,
        )

        if mask_time_indices:
            return output.loss, (
                output.extract_features, xs_len, output.last_hidden_state, xs_len
            )
        else:
            return output.extract_features, xs_len, output.last_hidden_state, xs_len

class SIMPLE_LINEAR_ENCODER(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        hidden_dims= [128, 128, 128, 128, 8],
        drop_out=0.2,
        norm_layer="batch",
        act={
            "select": "torch.nn:ReLU",
            "conf": {"inplace": True}, 
        },
        cmvn_path=None,
        load_from_h5=None,
    ):
        super().__init__()
        hidden_dims = copy.deepcopy(hidden_dims)
        hidden_dims.insert(0, feature_dim)
        num_layer = len(hidden_dims) - 1
        layers = OrderedDict()
        if cmvn_path == "SlidingWindow":
            logging.info(
                f"pre-sliding_window_cmn feature extractor in {self.__class__}"
            )
            self.feature_extractor = torchaudio.transforms.SlidingWindowCmn()
        elif cmvn_path:
            logging.info(f"pre-compute cmvn feature extractor in {self.__class__}")
            cmvn = torch.load(
                hydra.utils.to_absolute_path(cmvn_path),
                map_location=torch.device('cpu')
            )
            assert cmvn['normal']['mean'].size(-1) == feature_dim, \
                f"cmvn feature_dim not equal cmvn readin"
            self.feature_extractor = GlobalCMVN(
                mean=cmvn['normal']['mean'],
                istd=cmvn['normal']['var'],
            )
        else:
            logging.info(f"no feature extractor in {self.__class__}")
            self.feature_extractor = nn.Identity()
        for i in range(num_layer):
            layers[f'dropout{i}'] = nn.Dropout(p=drop_out, inplace=False)
            layers[f'linear{i}'] = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if norm_layer == "batch":
                layers[f'norm{i}'] = nn.BatchNorm2d(1)
            elif norm_layer == "audio":
                layers[f'norm{i}'] = AUDIO_NORM(hidden_dims[i+1])
            layers[f'activation{i}'] = init_object(act)
        self.net = nn.Sequential(layers)
        if load_from_h5 is not None:
            self.load_from_h5(load_from_h5)

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
                elif key.startswith('dense_'):
                    idx = int(key.split("_", 1)[1])
                    if idx <= 5:
                        conv2d_load(
                            net_weight[key][key],
                            getattr(self.enc, f"linear{idx-1}")
                        )
                        logging.warning(f"loading {key} to encoder linear{idx-1}")

    def forward(self, xs, xs_len):
        # x: (B, C, T, F)
        features = self.feature_extractor(xs)
        hs = self.net(features)
        return features, xs_len, hs, xs_len

class SIMPLE_LINEAR_VAR_ENCODER(SIMPLE_LINEAR_ENCODER):
    def __init__(
        self,
        feature_dim=128,
        hidden_dims=[128, 128, 128, 128, 8],
        drop_out=0.2,
        norm_layer="batch",
        act={
            "select": "torch.nn:ReLU",
            "conf": {"inplace": True}, 
        },
        cmvn_path=None,
        load_from_h5=None,
    ):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dims=hidden_dims[:-1],
            drop_out=drop_out,
            norm_layer=norm_layer,
            act=act,
            cmvn_path=cmvn_path,
            load_from_h5=load_from_h5,
        )
        self.fc_loc = nn.Linear(
            hidden_dims[-2], hidden_dims[-1]
        )
        self.fc_scale = nn.Linear(
            hidden_dims[-2], hidden_dims[-1]
        )
        self.latent_size = hidden_dims[-1]

    def forward(self, xs, xs_len):
        # x: (B, C, T, F)
        features = self.feature_extractor(xs)
        hs = self.net(features)
        z_loc = self.fc_loc(hs)
        z_scale = torch.exp(self.fc_scale(hs))
        return features, xs_len, hs, xs_len, z_loc, xs_len, z_scale, xs_len
    
    def get_latent_size(self):
        return self.latent_size
