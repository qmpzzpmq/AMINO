import torch
import pytorch_lightning as pl

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
    def data_extract(self, batch, feature_dim=None):
        return data_extract(batch, feature_dim)

    def data_seperation(self, batch, seperation_dim=0):
        return data_seperation(batch, seperation_dim=seperation_dim)
