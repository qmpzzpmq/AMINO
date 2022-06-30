import torch
import pyro
import pyro.distributions as dist

from AMINO.modules.base_module import ADMOS_MODULE, PYRO_PL_ADMOS_MODULE

# temp for ZHAOYI module
class AUTOENCODER_GMM(ADMOS_MODULE):
    def batch2loss(self, feature, feature_len):
        # feature shoule be (batch, channel, time, feature)
        _, feature_hat, z, gamma = self.net(feature)
        loss = self.losses(feature, feature_hat, z, gamma).sum()
        loss =  loss / feature_len.sum() / feature.size(-1)
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
        if "val_normal_loss" in losses and "val_anormal_loss" in losses:
            diff = losses["val_anormal_loss"] - losses["val_normal_loss"]
            self.log(
                f"val_diff_loss", diff,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True
            )
        return losses

class AMINO_AUTOENCODER_GMM(ADMOS_MODULE):
    def batch2loss(self, feature, feature_len):
        # feature shoule be (batch, channel, time, feature)
        _, feature_hat, z, gamma = self.net(feature, feature_len)
        loss = self.losses(feature, feature_hat, z, gamma).sum()
        loss =  loss / feature_len.sum() / feature.size(-1)
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
        if "val_normal_loss" in losses and "val_anormal_loss" in losses:
            diff = losses["val_anormal_loss"] - losses["val_normal_loss"]
            self.log(
                f"val_diff_loss", diff,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True
            )
        return losses

class PYRO_VAE_GMM(PYRO_PL_ADMOS_MODULE):
    def model(self, xs, xs_len):
        with pyro.plate("data", xs.shape[0], dim=-xs.dim()):
            # setup hyperparameters for prior p(z)
            xs_shape = list(xs.shape)
            xs_shape[-1] = self.net.encoder.get_latent_size()
            z_loc = torch.zeros(
                xs_shape, dtype=xs.dtype, device=xs.device
            )
            z_scale = torch.ones(
                xs_shape, dtype=xs.dtype, device=xs.device
            )
            # sample from prior (value will be sampled by guide when computing the ELBO)
            zs = pyro.sample(
                "latent", dist.Normal(z_loc, z_scale).to_event(1)
            )
            # decode the latent code z
            xs_hat, xs_hat_len = self.net.decoder.forward(zs, xs_len)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Bernoulli(xs_hat, validate_args=False).to_event(1),
                obs=xs,
            )
            # return the loc so we can visualize it later
            return xs_hat, xs_hat_len

    def guide(self, xs, xs_len):
        with pyro.plate("data", xs.shape[0], dim=-xs.dim()):
            # use the encoder to get the parameters used to define q(z|x)
            _, _, _, _, z_loc, _, z_scale, _ = self.net.encoder.forward(xs, xs_len)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def training_step(self, batch, batch_idx):
        seperated_batch = self.ADMOS_seperation(batch)
        feature = seperated_batch['normal']['data']
        feature_len = seperated_batch['normal']['len']
        loss = self.losses(
            self.model, self.guide,
            feature, feature_len
        )
        self.log(
            'loss', loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True,
        )
        self.optimizers
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        seperated_batch = self.ADMOS_seperation(batch)
        losses = dict()
        for key, value in seperated_batch.items():
            loss = self.losses(
                self.model, self.guide, 
                value['data'], value['len'],
            )
            self.log(
                f"val_{key}_loss", loss,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True
            )
            losses[f"val_{key}_loss"] = loss
        if "val_normal_loss" in losses and "val_anormal_loss" in losses:
            diff = losses["val_anormal_loss"] - losses["val_normal_loss"]
            self.log(
                f"val_diff_loss", diff,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True
            )
        return losses