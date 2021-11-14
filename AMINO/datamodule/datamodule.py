import logging

import torch.utils.data as tdata
import pytorch_lightning as pl

from AMINO.datamodule.datasets import init_datasets
from AMINO.datamodule.preprocess import init_preporcesses
from AMINO.utils.datamodule import AMINOPadCollate

class AMINODataModule(pl.LightningDataModule):
    def __init__(self, datamodule_conf):
        super().__init__()
        self.datamodule_conf = datamodule_conf
        self.save_hyperparameters()
        self.datasets = dict()
        self.collect_fns = dict()
        self.transform = {'after': dict()}
        self.transform2device = {'after': dict()}
        for dataset in ['train', 'val', 'test']:
            self.transform['after'][dataset] = self.preprocess_get(
                'after_transform', dataset
            )
            self.transform2device['after'][dataset] = False

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
        # download
        # tokenize
        # etc…
        super().prepare_data()

    def setup(self, stage=None):
        # There are also data operations you might want to perform on every GPU. Use setup to do things like:
        # count number of classes
        # build vocabulary
        # perform train/val/test splits
        # apply transforms (defined explicitly in your datamodule)
        # etc…
        if stage in (None, 'fit'):
            self.stage_setup(['train', 'val'])

        if stage in (None, 'test'):
            self.stage_setup(['test'])

    def stage_setup(self, key_selects):
        precrocesses = dict()
        for key, value in self.datamodule_conf['single_preprocesses'].items():
            if key in key_selects:
                precrocesses[key] = init_preporcesses(value)
            for key, value in self.datamodule_conf['datasets'].items():
                if key in key_selects:
                    self.datasets[key] = init_datasets(value)
                    if self.datasets[key] is not None:
                        self.datasets[key].set_preprocesses(
                            precrocesses.get(key, None)
                        )
                    self.collect_fns[key] = AMINOPadCollate()

    def train_dataloader(self):
        if self.datasets['train'] is not None:
            return tdata.DataLoader(
                self.datasets['train'],
                **self.datamodule_conf['dataloaders']['train'],
                collate_fn=self.collect_fns['train'],
            )
        else:
            return None

    def val_dataloader(self):
        if self.datasets['val'] is not None:
            return tdata.DataLoader(
                self.datasets['val'],
                **self.datamodule_conf['dataloaders']['val'],
                collate_fn=self.collect_fns['val'],
            )
        else:
            return None

    def test_dataloader(self):
        if self.datasets['test'] is not None:
            return tdata.DataLoader(
                self.datasets['test'],
                **self.datamodule_conf['dataloaders']['test'],
                collate_fn=self.collect_fns['test']
            )
        else:
            return None

    # this bug fix: when pytorch_lightning > 1.5
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     if type(self.trainer.accelerator) == pl.accelerators.cpu.CPUAccelerator:
    #         print("0")
    #         return self.on_after_batch_transfer(batch, dataloader_idx)
    #     print(f"1. accelerator: {type(self.trainer.accelerator)}")
    #     return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training \
                and self.transform['after'].get('train', None) is not None:
            return self.batch_transform('after', 'train', batch)
        elif (self.trainer.validating or self.trainer.sanity_checking) \
                and self.transform['after'].get('val', None) is not None:
            return self.batch_transform('after', 'val', batch)
        elif self.trainer.testing \
                and self.transform['after'].get('test', None) is not None:
            return self.batch_transform('after', 'test', batch)
        elif self.trainer.predicting \
                and self.transform['after'].get('predict', None) is not None:
            return self.batch_transform('after', 'predict', batch)
        return batch

    def batch_transform(self, position, key, batch):
        if not self.transform2device[position][key]:
            self.transform[position][key] = \
                self.transform[position][key].to(
                    batch['feature']['data'].device
                )
        batch = self.transform[position][key](batch)
        return batch

    def preprocess_get(self, position, key):
        if position in self.datamodule_conf and \
                key in self.datamodule_conf[position]:
            return init_preporcesses(
                self.datamodule_conf[position][key],
            )
        else:
            return None

    def tesrdown(self):
        super().teardown()

def init_datamodule(datamodule_conf):
    return AMINODataModule(datamodule_conf)
