import torch

from AMINO.modules.base_module import AMINO_MODULE

class AMINO_CLASSIFIER(AMINO_MODULE):
    def batch2loss(self, batch):
        pred_dict, pred_len_dict = self.net(
            batch['feature']['data'].squeeze(1),
            batch['feature']['len']
        )
        loss_dict = dict()
        total_loss = torch.tensor(0.0, device=batch['feature']['data'].device)
        for k in pred_dict.keys():
            if k.startswith("classifier"):
                loss = self.losses[k](
                    pred_dict[k], batch['label']['data']
                )
                loss_dict[k] = loss.sum()
                total_loss += loss_dict[k] * self.losses_weight[k]
            elif k.startwith("autoencoder"):
                loss = self.losses[k](
                    pred_dict[k], batch['feature']['data']
                )
                loss_dict[k] = loss.sum()
                total_loss += loss_dict[k] * self.losses_weight[k]
        loss_dict['total'] = total_loss 
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.batch2loss(batch)
        for k, v in loss_dict.items():
            self.log(
                f'loss_{k}', v,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True,
            )
        return {'loss': loss_dict['total']}

    def validation_step(self, batch, batch_idx):
        loss_dict = self.batch2loss(batch)
        for k, v in loss_dict.items():
            self.log(
                f"val_loss_{k}", v,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True,
            )
    