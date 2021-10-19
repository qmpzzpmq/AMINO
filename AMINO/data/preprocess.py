from abc import ABC

import torch
import torch.nn as nn
import torchaudio

class AUDIO_GENERAL(nn.Module):
    def __init__(self, mono_channel, fs):
        super().__init__()
        self.fs = fs
        if mono_channel == "mean":
            self.mono_func = lambda x: torch.mean(x, dim=0)
        elif mono_channel.isdigit() and len(mono_channel) == 1:
            self.mono_func = lambda x: x[int(mono_channel), :]
        else:
            raise ValueError(f"the mono_channel setting wrong, please read the help in AMINO/configs/datamodule.py")
    def forward(self, data, fs):
        data = self.mono_func(data)
        data = torchaudio.functional.resample(data, fs, self.fs)
        return data, fs

class FFT(nn.Module):
    def __init__(self, fft_conf):
        super().__init__()
        self.fft = torchaudio.transforms.Spectrogram(**fft_conf, power=2)

    def forward(self, data, fs):
        # data shape should be [channel, time]
        data_pwr = self.fft(data) 
        return data_pwr, fs

def init_preporcesses(preprocesses_conf):
    preprocesses = []
    for preprocess in preprocesses_conf:
        preprocess_class = eval(preprocess['select'])
        preprocess = preprocess_class(**preprocess['conf'])
        preprocesses.append(preprocess)
    return torch.nn.Sequential(preprocesses)