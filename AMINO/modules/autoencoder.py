from AMINO.modules.nets.nets import init_net
from AMINO.modules.loss import init_loss
from AMINO.modules.optim import init_optim
from AMINO.modules.scheduler import init_scheduler
from AMINO.modules.base_module import AMINO_MODULE

class AMINO_AUTOENCODER(AMINO_MODULE):
    def __init__(self, net_conf, loss_conf, optim_conf=None, scheduler_conf=None):
        super().__init__()
        self.net = init_net(net_conf)
        self.loss = init_loss(loss_conf)
        if optim_conf is not None:
            self.optim = init_optim(self.net, optim_conf)
            if scheduler_conf is not None:
                self.scheduler = init_scheduler(self.optim, scheduler_conf)
        self.save_hyperparameters()

    def batch2loss(self, feature, datas_len, reduction='feature_len'):
        pred = self.net(feature)
        reduction_len = datas_len[0].sum() \
            if reduction=='feature_len' else datas_len[1].sum()
        loss = self.loss(pred, feature).sum() / reduction_len
        return loss

    def training_step(self, batch, batch_idx):
        feature, label, datas_len = self.data_extract(batch)
        loss = self.batch2loss(feature, datas_len)
        self.log(
            'train_loss', loss,
            on_step=True, on_epoch=True, 
            prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feature, label, datas_len = self.data_extract(batch)
        loss = self.batch2loss(feature, datas_len)
        self.log(
            'val_loss', loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]