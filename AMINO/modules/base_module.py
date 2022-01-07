import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from AMINO.utils.init_object import init_object
from AMINO.modules.optim import init_optim
from AMINO.modules.scheduler import init_scheduler


def data_seperation(batch, seperation_dim=0):
    feature = batch['feature']['data']
    feature_len = batch['feature']['len']
    label = batch['label']['data']
    out_dict = dict()
    for key, flag in zip(["normal", "anomaly"], [True, False]):
        idx = (label==flag).nonzero(as_tuple=True)[0]
        if idx.size(0) > 0:
            temp_feature = torch.index_select(
                feature, seperation_dim, idx
            )
            temp_feature_len = torch.index_select(
                feature_len, seperation_dim, idx
            )
            out_dict[key] = {
                "data": temp_feature, "len": temp_feature_len
            }
    return out_dict

class AMINO_MODULE(pl.LightningModule):
    def __init__(self,
            net=None, 
            loss=None,
            optim=None,
            scheduler=None,  
    ):
        super().__init__()
        self.net = init_object(net)
        if loss:
            self.loss = init_object(loss)
        if optim:
            self.optim = init_optim(self.net, optim)
            if scheduler:
                self.scheduler = init_scheduler(self.optim, scheduler)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]

class ADMOS_MODULE(AMINO_MODULE):
    def data_seperation(self, batch, seperation_dim=0):
        return data_seperation(batch, seperation_dim=seperation_dim)

    def feature_statistics_init(self, feature_dim):
        self.avg_features = nn.ParameterDict({
            "normal_mean": nn.Parameter(
                torch.zeros([1, feature_dim]), 
                False
            ),
            "anomaly_mean": nn.Parameter(
                torch.zeros([1, feature_dim]), 
                False
            ),
            "normal_var": nn.Parameter(
                torch.zeros([1, feature_dim]), 
                False
            ),
            "anomaly_var": nn.Parameter(
                torch.zeros([1, feature_dim]), 
                False
            ),
            "normal_num": nn.Parameter(
                torch.zeros([1, feature_dim]), 
                False
            ),
            "anomaly_num": nn.Parameter(
                torch.zeros([1, feature_dim]), 
                False
            ),
        })

    # should be predict_step
    def feature_statistics(self, batch, batch_idx):
        avg_features = OrderedDict()
        features, features_len = self.data_seperation(batch)
        # avg_feature[key]: (batch, channel, time, feature)
        for key in features.keys():
            avg_features[f'{key}_mean'] = nn.Parameter(
                self.avg_features[f'{key}_mean'] 
                    + features[key].sum(dim=(0, 2)),
                False,
            )
            avg_features[f'{key}_var'] = nn.Parameter(
                self.avg_features[f'{key}_mean'] 
                    + features[key].square().sum(dim=(0, 2)),
                False,
            )
            avg_features[f'{key}_num'] = nn.Parameter(
                self.avg_features[f'{key}_num'] + features_len[key].sum(),
                False,
            )
        self.avg_features.update(avg_features)
        
    def feature_statistics_end(self, dump_path=None):
        cmvn = OrderedDict()
        for key in ['normal', 'anomaly']:
            cmvn[key] = OrderedDict()
            cmvn[key]['mean'] = self.avg_features[f'{key}_mean'] / self.avg_features[f'{key}_num']
            cmvn[key]['var'] = (
                self.avg_features[f'{key}_mean'] / self.avg_features[f'{key}_num']
                - cmvn[key]['mean'] **2 
            ).clamp(min=1.0e-20)
            cmvn[key]['var'] = 1.0 / cmvn[key]['var'].sqrt()
            logging.info(f"{key} mean feature is {cmvn[key]['mean']}")
            logging.info(f"{key} var features is {cmvn[key]['var']}")
        if dump_path is not None:
            logging.warning(f"dump state dict to {dump_path}")
            torch.save(cmvn, dump_path)
        for part in ['mean', 'var']:
            diff = cmvn['normal'][part] - cmvn['anomaly'][part]
            logging.warning(f"the diff {part} of two feature is {diff}")
        logging.warning(
            f"the average of normal is {cmvn['normal']['mean'].mean()}"
        )
        logging.warning(
            f"the average of normal is {cmvn['anomaly']['mean'].mean()}"
        )
        logging.warning(f"the average of diff is {diff.mean()}")
        del self.avg_features
