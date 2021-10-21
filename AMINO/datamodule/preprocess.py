from abc import ABC

import torch
import torch.nn as nn
import torchaudio

class AUDIO_GENERAL(nn.Module):
    # right now only avaiable at single_preprocess
    # should be the first be single_preprocess
    def __init__(self, mono_channel, fs):
        super().__init__()
        self.fs = fs
        if mono_channel == "mean":
            self.mono_func = lambda x: torch.mean(x, dim=0).unsqueeze(0)
        elif mono_channel.isdigit() and len(mono_channel) == 1:
            self.mono_func = lambda x: x[int(mono_channel), :].unsqueeze(0)
        else:
            raise ValueError(
                f"the mono_channel setting wrong, please read the help in AMINO/configs/datamodule.py"
            )

    def forward(self, ins):
        # data.shape: [channel, time]
        data, fs = ins
        data = self.mono_func(data)
        if fs != self.fs:
            data = torchaudio.functional.resample(data, fs, self.fs)
        return data

class FFT(nn.Module):
    def __init__(self, **fft_conf):
        super().__init__()
        self.fft = torchaudio.transforms.Spectrogram(**fft_conf, power=2)

    def forward(self, ins):
        datas, datas_len = ins
        datas_pwr = self.fft(datas).transpose(-1, -2)
        # right now, datas_len not correct
        return [datas_pwr, datas_len]

def init_preporcesses(preprocesses_conf):
    if preprocesses_conf is not None:
        preprocesses = []
        for preprocess in preprocesses_conf:
            preprocess_class = eval(preprocess['select'])
            preprocess = preprocess_class(**preprocess['conf'])
            preprocesses.append(preprocess)
        return torch.nn.Sequential(*preprocesses)
    else:
        return None