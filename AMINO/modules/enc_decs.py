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
            # temp for debug
            loss_dict, feature, pred_dict, pred_len_dict = self.batch2loss(batch)
            # logging.info(
            #     f"{torch.distributed.get_rank()}| train batch idx {batch_idx} get loss {loss_dict}"
            # )
        except Exception as e1:
            with torch.no_grad():
                logging.warning(f"something wrong: {e1}")
                check_result = total_check(batch, dim=1)
                save_error_tesnsor(batch, os.getcwd())
                torch.cuda.empty_cache()
            logging.warning("batch failed, try to use single item as one batch")
            batch_size = batch['feature']['data'].size(0)
            for idx in range(batch_size):
                try:
                    loss_dict, feature, pred_dict, pred_len_dict = self.batch2loss(
                        self.batch_select(batch, idx)
                    )
                except Exception as e2:
                    logging.warning(f"{idx}item as batch failed with {e2}")
                else:
                    logging.warning(
                        f"{idx}item as batch succeed, loss: {loss_dict}"
                    )
                    return {'loss': loss_dict['total']}
            raise ValueError("all item in this batch failed")
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/5243
            # return None
            # https://github.com/pytorch/pytorch/issues/23425
            # return {
            #     'loss': torch.tensor(
            #         0.0, device=batch['label']['data'].device,
            #     )
            # }
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

class AMINO_WAC2VEC_ENC_DECS(AMINO_ENC_DECS):
    def batch2loss(self, batch):
        wav2vec2_loss, casual_output = self.net(
            batch['feature']['data'],
            batch['feature']['len'],
        )
        feature, feature_len, pred_dict, pred_len_dict = casual_output
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
        loss_dict["wav2vec2"] = wav2vec2_loss
        return loss_dict, feature, pred_dict, pred_len_dict