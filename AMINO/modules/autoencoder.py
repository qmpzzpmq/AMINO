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

    def batch2loss(self, feature, feature_len):
        # feature shoule be (batch, channel, time, feature)
        pred = self.net(feature)
        loss = self.loss(pred, feature).sum()
        loss =  loss / feature_len.sum() / pred.size(-1)
        return loss

    def training_step(self, batch, batch_idx):
        feature, label, datas_len = self.data_extract(batch)
        loss = self.batch2loss(feature, datas_len[0])
        self.log(
            'loss', loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True,
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        features, features_len = self.data_seperation(batch)
        losses = dict()
        for key in features.keys():
            loss = self.batch2loss(features[key], features_len[key])
            self.log(
                f"val_{key}_loss", loss,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True
            )
            losses[f"val_{key}_loss"] = loss
        return losses

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]