import os
import logging

import torch

from AMINO.modules.base_module import AMINO_MODULE
from AMINO.utils.data_check import total_check, save_error_tesnsor

class AMINO_ENC_DECS(AMINO_MODULE):
    def __init__(
            self,
            net=None, 
            losses=None,
            optim=None,
            scheduler=None,
            metrics=None,
        ):
            super().__init__(net, losses, optim, scheduler, metrics)
            if self.losses:
                assert list(self.losses.nets.keys()) == list(self.net.decoders.keys())

    def batch2loss(self, batch):
        feature, feature_len, pred_dict, pred_len_dict = self.net(
            batch['feature']['data'],
            batch['feature']['len']
        )
        loss_dict = self.losses(
            pred_dict,
            {
                "classifier": batch['label']['data'],
                "autoencoder": feature,
            },
            {
                "classifier": batch["label"]["len"].sum(),
                "autoencoder": batch['feature']['len'].sum(),
            },
        )
        return loss_dict, feature, pred_dict, pred_len_dict

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        try:
            loss_dict, feature, pred_dict, pred_len_dict = self.batch2loss(batch)
        except Exception as e:
            with torch.no_grad():
                logging.warning(f"something wrong: {e}")
                check_result = total_check(batch, dim=1)
                save_error_tesnsor(batch, os.getcwd())
                torch.cuda.empty_cache()
                return None
        for k, v in loss_dict.items():
            self.log(
                f'loss_{k}', v,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True,
            )
        return {'loss': loss_dict['total']}

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        loss_dict, feature, pred_dict, pred_len_dict = self.batch2loss(batch)
        # try:
        #     loss_dict, feature, pred_dict, pred_len_dict = self.batch2loss(batch)
        # except Exception as e:
        #     logging.warning(f"something wrong: {e}")
        #     check_result = total_check(batch, dim=2)
        #     save_error_tesnsor(batch, os.getcwd())
        #     return None
        for k, v in loss_dict.items():
            self.log(
                f"val_loss_{k}", v,
                on_step=True, on_epoch=True,
                prog_bar=True, logger=True,
            )

    def log_metrics(
            self,
            batch,
            feature=None,
            feature_len=None,
            pred_dict=None,
            pred_len_dict=None,
        ):
        if not (
                (pred_dict is None) or 
                (pred_len_dict is None) or 
                (feature is None) or 
                (feature_len is None)
            ):
            feature, feature_len, pred_dict, pred_len_dict = self.net(
                batch['feature']['data'].squeeze(1),
                batch['feature']['len']
            )
        for pred_name, pred_obj in pred_dict.keys():
            if pred_name == "classifier":
                for metric_name, metric_obj \
                        in self.metrics["classifier"].items():
                    metric_obj(pred_obj, batch['label']['data'])
                    self.log(f"{pred_name}-{metric_name}", metric_obj)
            elif pred_name == "autoencoder":
                for metric_name, metric_obj \
                        in self.metrics["classifier"].items():
                    metric_obj(pred_obj, batch['label']['data'])
                    self.log(f"{pred_name}-{metric_name}", metric_obj)

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
        try:
            self.log_metrics(batch)
        except Exception as e:
            logging.warning(f"something wrong: {e}")
            check_result = total_check(batch, dim=2)
            save_error_tesnsor(batch, os.getcwd())
            return None
        