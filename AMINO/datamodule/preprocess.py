import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from AMINO.utils.dynamic_import import dynamic_import

class TrainDataAugment(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class Spectrogram(nn.Module):
    def __init__(self, **fft_conf):
        super().__init__()
        self.fft = torchaudio.transforms.Spectrogram(**fft_conf)

    def forward(self, batch):
        feature = batch['feature']['data']
        feature_len = batch['feature']['len']
        feature_pwr = self.fft(feature).transpose(-1, -2)
        feature_len = (
            (feature_len - self.fft.n_fft) / self.fft.hop_length + 3
        ).floor().to(torch.int32)
        batch['feature'] = {'data': feature_pwr, 'len': feature_len}
        return batch

class MelSpectrogram(nn.Module):
    def __init__(self, **mel_conf):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(**mel_conf)

    def forward(self, batch):
        feature = batch['feature']['data']
        feature_len = batch['feature']['len']
        melspec = self.melspec(feature).transpose(-1, -2)
        feature_len = (
            (feature_len - self.melspec.n_fft) / self.melspec.hop_length + 3
        ).floor().to(torch.int32)
        batch['feature'] = {'data': melspec, 'len': feature_len}
        return batch

# modified from https://github.com/espnet/espnet/pull/2324
class Feature_Unfold(nn.Module):
    def __init__(
            self, 
            n_frame=5,
        ):
        super().__init__()
        self.n_frame = n_frame

    def forward(self, batch):
        feature = batch['feature']['data']
        b, c, t, f = feature.shape
        feature = F.pad(
            feature, pad=(0, 0, self.n_frame - 1, 0), value=0.0
        )
        # feature: (batch, channel, time, feature) ->
        # (batch, channel, time+self.n_frame-1, feature)
        feature = F.unfold(feature, (self.n_frame, 1)).view(
            b, c, self.n_frame, t, f
        ).transpose(2,3).reshape(b, c, t, -1)
        # feature: (batch, channel*n_frame, fime*feature) ->
        # (batch, channel, n_frame, Time, feature) ->
        # (batch, channel, Time, n_frame, feature) ->
        # (batch, channel, Time, n_frame * feature) 
        batch['feature']['data'] = feature
        return batch

# should be imporve later for T for a retio of length
class SpecAug(TrainDataAugment):
    def __init__(self, **specaug_conf):
        super().__init__()
        self.num_mask = dict()
        if 'frequency_mask' in specaug_conf:
            self.FM = torchaudio.transforms._AxisMasking(
                mask_param=specaug_conf['frequency_mask'].get("F", 30),
                axis = 2,
                iid_masks=True,
            )
            self.num_mask['frequency'] = specaug_conf['frequency_mask']["num_mask"]
        else:
            self.FM = None
        if 'time_mask' in specaug_conf:
            self.TM = torchaudio.transforms._AxisMasking(
                mask_param=specaug_conf['time_mask'].get("T", 40),
                axis = 1,
                iid_masks=True,
            )
            self.num_mask['time'] = specaug_conf['time_mask']["num_mask"]
        else:
            self.TM = None
        if "time_stretch" in specaug_conf:
            self.time_stretch_range = [
                specaug_conf['time_mask'].get('floor', 0.9),
                specaug_conf['time_mask'].get('ceil', 1.1),
            ]
            self.TS = torchaudio.transforms.TimeStretch()
        else:
            self.time_stretch_range = None

    def forward(self, batch):
        feature = batch['feature']['data']
        # feature_pwr: (batch, channel, time, feature)
        if self.time_stretch_range is not None:
            feature = self.TS(
                feature.transpose(-1, -2),
                random.uniform(*self.time_stretch_range)
            ).transpose(-1, -2)
        if self.FM is not None:
            for _ in range(self.num_mask['frequency']):
                feature = self.FM(feature)
        if self.TM is not None:
            for _ in range(self.num_mask['time']):
                feature = self.TM(feature)
        batch['feature']['data'] = feature
        return batch

def init_preporcesses(preprocesses_conf):
    if preprocesses_conf is not None:
        preprocesses = []
        for preprocess in preprocesses_conf:
            preprocess_class = dynamic_import(preprocess['select'])
            preprocess = preprocess_class(**preprocess['conf'])
            preprocesses.append(preprocess)
        return torch.nn.Sequential(*preprocesses)
    else:
        return None