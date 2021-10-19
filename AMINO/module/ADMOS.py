import torch
import torch.nn as nn

from AMINO.module.frontend import init_frontend
from AMINO.module.model import init_model
from AMINO.module.loss import init_loss
from AMINO.module.optim import init_optim

class ADMOS(nn.module):
    def __init__(self, frontend_conf, model_conf, loss_conf, optim_conf):
        self.frontend = init_frontend(frontend_conf)
        self.model = init_model(model_conf)
        self.loss = init_loss(loss_conf)
        self.optim = init_optim(optim_conf)

    def forward(self, datas, datas_len):
        self.frontend(datas, dats_len)

def init_module(model_cfg):
    