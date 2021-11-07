from AMINO.modules.base_module import AMINO_MODULE

class AMINO_AUTOENCODER(AMINO_MODULE):
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