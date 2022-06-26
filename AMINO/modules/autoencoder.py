from AMINO.modules.base_module import ADMOS_MODULE

class AMINO_AUTOENCODER(ADMOS_MODULE):
    def batch2loss(self, feature, feature_len):
        # feature shoule be (batch, channel, time, feature)
        pred = self.net(feature)
        loss = self.loss(pred, feature).sum()
        loss =  loss / feature_len.sum() / pred.size(-1)
        return loss

    def training_step(self, batch, batch_idx):
        seperated_batch = self.ADMOS_seperation(batch)
        feature = seperated_batch['normal']['data']
        feature_len = seperated_batch['normal']['len']
        loss = self.batch2loss(feature, feature_len)
        self.log(
            'loss', loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True,
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        seperated_batch = self.ADMOS_seperation(batch)
        losses = dict()
        for k, v in seperated_batch.items():
            loss = self.batch2loss(v['data'], v['len'])
            self.log(
                f"val_{k}_loss", loss,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True
            )
            losses[f"val_{k}_loss"] = loss
        return losses

class AMINO_AUTOENCODER(ADMOS_MODULE):
    def batch2loss(self, feature, feature_len):
        # feature shoule be (batch, channel, time, feature)
        _, feature_hat, z, gamma = self.net(feature)
        loss = self.losses(feature, feature_hat, z, gamma).sum()
        loss =  loss / feature_len.sum() / feature_hat.size(-1)
        return loss

    def training_step(self, batch, batch_idx):
        seperated_batch = self.ADMOS_seperation(batch)
        feature = seperated_batch['normal']['data']
        feature_len = seperated_batch['normal']['len']
        loss = self.batch2loss(feature, feature_len)
        self.log(
            'loss', loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True,
        )
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        seperated_batch = self.ADMOS_seperation(batch)
        losses = dict()
        for k, v in seperated_batch.items():
            loss = self.batch2loss(v['data'], v['len'])
            self.log(
                f"val_{k}_loss", loss,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True
            )
            losses[f"val_{k}_loss"] = loss
        return losses