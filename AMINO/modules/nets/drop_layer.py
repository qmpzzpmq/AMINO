import torch
import torch.nn as nn

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer

class DROP_LAYER(nn.Module):
    def __init__(self, p, layer):
        super().__init__()
        self.p = p
        self.layer = layer

    def forward(self, xs, xs_len):
        if self.training and torch.rand(1)[0] > self.p:
            return self.layer(xs, xs_len)
        else:
            return xs, xs_len

class Wav2Vec2EncoderLayer_DropLayer(Wav2Vec2EncoderLayer):
    def __init__(self, layer_drop_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_drop_p = layer_drop_p

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if self.training and torch.rand(1)[0] > self.p:
            return super().forward(
                hidden_states, attention_mask, output_attentions
            )
        else:
            return hidden_states

class Wav2Vec2EncoderLayer_DropLayer_Warraper(DROP_LAYER):
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if self.training and torch.rand(1)[0] > self.p:
            return super().forward(
                hidden_states, attention_mask, output_attentions
            )
        else:
            return hidden_states