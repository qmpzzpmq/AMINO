import pytorch_lightning as pl

def data_extract(batch):
    datas, datas_len = batch
    feature, label = datas
    return feature, label, datas_len

def data_pack(feature, label, datas_len):
    return [feature, label], datas_len

class AMINO_MODULE(pl.LightningModule):
    def data_extract(self, batch):
        return data_extract(batch)