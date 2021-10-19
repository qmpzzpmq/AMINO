import torch.utils.data as tdata
import pytorch_lightning as pl

from AMINO.data.datasets import init_dataset
from AMINO.data.preprocess import init_preporcesses
from AMINO.data.utils import SinglePadCollate

class AMINODataModule(pl.LightningDataModule):
    def __init__(self, datamodule_conf):
        self.datamodule_conf = datamodule_conf

    def prepare_data(self):
        self.datasets = dict()
        self.collect_futs = dict()

    def setup(self, stage=None):
        precrocesses = dict()
        if stage in (None, 'fit'):
            key_selects = ['train', 'dev']
            for key, value in self.datamodule_conf['preprocess']:
                if key in key_selects:
                    precrocesses[key] = init_preporcesses(value)

            for key, value in self.datamodule_conf['datasets']:
                if key in key_selects:
                    self.datasets[key] = init_dataset(value)
                    self.datasets[key].set_preprocesses(precrocesses[key])
                    self.collect_futs[key] = SinglePadCollate()

        if stage in (None, 'test'):
            key_selects = ['test']
            for key, value in self.datamodule_conf['preprocess']:
                if key in key_selects:
                    precrocesses[key] = init_preporcesses(value)
            for key, value in self.datamodule_conf['datasets']:
                if key in key_selects:
                    self.datasets[key] = init_dataset(value)
                    self.datasets[key].set_preprocesses(precrocesses[key])
                    self.collect_futs = SinglePadCollate()

    def train_dataloader(self):
        return tdata.DataLoader(
            self.dataset['train'], **self.datamodule_conf['dataloaders']['train'],
            collate_fn=self.collect_futs['train']
        )

    def val_dataloader(self):
        return tdata.DataLoader(
            self.dataset['dev'], **self.datamodule_conf['dataloaders']['dev'],
            collate_fn=self.collect_futs['dev']
        )

    def test_dataloader(self):
        return tdata.DataLoader(
            self.dataset['test'], **self.datamodule_conf['dataloaders']['test'],
            collate_fn=self.collect_futs['test']
        )

def init_datamodule(datamodule_conf):
    return AMINODataModule(datamodule_conf)
