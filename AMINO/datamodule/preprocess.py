import torch
import torch.nn as nn
import torchaudio

from AMINO.modules.base_module import data_extract, data_pack
from AMINO.utils.dynamic_import import dynamic_import

class FFT(nn.Module):
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