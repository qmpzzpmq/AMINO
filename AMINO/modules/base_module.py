import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from AMINO.modules.nets.nets import init_net
from AMINO.modules.loss import init_loss
from AMINO.modules.optim import init_optim
from AMINO.modules.scheduler import init_scheduler

def data_extract(batch, feature_dim=None):
    datas, datas_len = batch
    feature, label = datas
    return feature, label, datas_len

def data_pack(feature, label, datas_len):
    return [feature, label], datas_len

def data_seperation(batch, seperation_dim=0):
    datas, datas_len = batch
    feature, label = datas
    feature_len, label_len = datas_len
    out_feature = dict()
    out_feature_lens = dict()
    for key, flag in zip(['normal', 'anomaly'], [True, False]):
        idx = (label==flag).nonzero(as_tuple=True)[0]
        if idx.size(0) > 0:
            out_feature[key] = torch.index_select(
                feature, seperation_dim, idx
            )
            out_feature_lens[key] = torch.index_select(
                feature_len, seperation_dim, idx
            )
    return out_feature, out_feature_lens

def data_systhetic(
        normal_feature, anormal_feature,
        label, datas_len, synthetic_dim=0
    ):
    feature = torch.cat([normal_feature, anormal_feature], dim=synthetic_dim)
    return data_pack(feature, label, datas_len)

class AMINO_MODULE(pl.LightningModule):
    def __init__(self, 
            net_conf=None, 
            loss_conf=None,
            optim_conf=None,
            scheduler_conf=None,
            cmvn=False,    
    ):
        super().__init__()
        if net_conf:
            self.net = init_net(net_conf)
        if loss_conf:
            self.loss = init_loss(loss_conf)
        if optim_conf is not None:
            self.optim = init_optim(self.net, optim_conf)
            if scheduler_conf is not None:
                self.scheduler = init_scheduler(self.optim, scheduler_conf)
        self.save_hyperparameters()

    def data_extract(self, batch, feature_dim=None):
        return data_extract(batch, feature_dim)

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
