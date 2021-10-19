import torch
import torch.nn as nn
import torchaudio as ta

class FRONTEND(nn.Module):
    def __init__(self, fft_conf, agc_conf):
        super().__init__()
        self.fft = ta.transforms.Spectrogram(**fft_conf, power=2)

    def forward(self, datas, datas_len):
        datas_pwr = self.fft = datas
        return datas_pwr

def init_frontend(frontend_conf):
    return FRONTEND(frontend_conf['fft'])