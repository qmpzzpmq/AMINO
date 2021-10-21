import torch.nn as nn
import pytorch_lightning as pl

from AMINO.modules.nets.nets import init_net
from AMINO.modules.loss import init_loss
from AMINO.modules.optim import init_optim
from AMINO.modules.scheduler import init_scheduler

class AMINO_AUTOENCODER(pl.LightningModule):
    def __init__(self, net_conf, loss_conf, optim_conf=None, scheduler_conf=None):
        super().__init__()
        self.net = init_net(net_conf)
        self.loss = init_loss(loss_conf)
        if optim_conf is not None:
            self.optim = init_optim(self.net, optim_conf)
            if scheduler_conf is not None:
                self.scheduler = init_scheduler(self.optim, scheduler_conf)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        datas, datas_len = batch
        feature = datas[0]
        pred = self.net(feature)
        loss = self.loss(pred, feature).sum()/datas_len.sum()
        self.log(
            'train_loss', loss,
            on_step=True, on_epoch=True, 
            prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        datas, datas_len = batch
        feature = datas[0]
        pred = self.net(feature)
        loss = self.loss(pred, feature).sum() / datas_len
        self.log(
            'val_loss', loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]