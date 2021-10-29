import random

import torch
import torch.nn as nn
import torchaudio

from AMINO.modules.base_module import data_extract, data_pack
from AMINO.utils.dynamic_import import dynamic_import

class Spectrogram(nn.Module):
    def __init__(self, **fft_conf):
        super().__init__()
        self.fft = torchaudio.transforms.Spectrogram(**fft_conf)

    def forward(self, batch):
        feature, label, datas_len = data_extract(batch)
        feature_pwr = self.fft(feature).transpose(-1, -2)
        datas_len[0] = (
            (datas_len[0] - self.fft.n_fft) / self.fft.hop_length + 3
        ).floor().to(torch.int32)
        return data_pack(feature_pwr, label, datas_len)

class MelSpectrogram(nn.Module):
    def __init__(self, **mel_conf):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(**mel_conf)
    def forward(self, batch):
        feature, label, datas_len = data_extract(batch)
        melspec = self.melspec(feature).transpose(-1, -2)
        datas_len[0] = (
            (datas_len[0] - self.melspec.n_fft) / self.melspec.hop_length + 3
        ).floor().to(torch.int32)
        return data_pack(melspec, label, datas_len)

# should be imporve later for T for a retio of length
class SpecAug(nn.Module):
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
        feature_pwr, label, datas_len = data_extract(batch)
        # feature_pwr: (batch, channel, time, feature)
        if self.time_stretch_range is not None:
            feature_pwr = self.TS(
                feature_pwr.transpose(-1, -2),
                random.uniform(*self.time_stretch_range)
            ).transpose(-1, -2)
        if self.FM is not None:
            for _ in range(self.num_mask['frequency']):
                feature_pwr = self.FM(feature_pwr)
        if self.TM is not None:
            for _ in range(self.num_mask['time']):
                feature_pwr = self.TM(feature_pwr)
        return data_pack(feature_pwr, label, datas_len)

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