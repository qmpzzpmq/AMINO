import torch.nn as nn
import pytorch_lightning as pl

from AMINO.module.nets import init_nets
from AMINO.module.loss import init_loss
from AMINO.module.optim import init_optim
from AMINO.module.scheduler import init_scheduler

class AMINO_AUTOENCODER(pl.LightningModule):
    def __init__(self, net_conf, loss_conf, optim_conf):
        super().__init__()
        self.net = init_nets(net_conf)
        self.loss = init_loss(loss_conf)
        if optim_conf is not None:
            self.optim = init_optim(self.raaec, optim_conf['optim'])
            self.scheduler = init_scheduler(self.optim, optim_conf['scheduler'])
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        feature, target = batch
        pred = self.net(feature)
        loss = self.loss(pred, target)
        self.log(
            'train_loss', loss,
            on_step=True, on_epoch=True, 
            prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feature, target = batch
        pred = self.net(feature)
        loss = self.loss(pred, target)
        self.log(
            'val_loss', loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]