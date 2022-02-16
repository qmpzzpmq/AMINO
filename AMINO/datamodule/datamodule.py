import logging

import torch
import torch.utils.data as tdata
import pytorch_lightning as pl

from AMINO.datamodule.preprocess import init_preporcesses
from AMINO.utils.datamodule import AMINOPadCollate
from AMINO.datamodule.datasets import AMINO_ConcatDataset
from AMINO.utils.init_object import init_list_object

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
    
    def get_num_classes(self):
        if hasattr(self, "num_classes"):
            return self.num_classes
        try:
            num_classes = torch.tensor([
                dataset.get_num_classes() 
                for dataset in self.datasets.values() if dataset is not None
            ])
            assert num_classes.max() == num_classes.min(), \
                f"num_classes in each data is not same"
            num_classes = num_classes.max().item()
        except Exception as e:
            logging.warning(
                f"Something wrong with get_num_classes, return None"
            )
            num_classes = None
        self.num_classes = num_classes
        return self.num_classes

    def stage_setup(self, key_selects):
        precrocesses = dict()
        if "single_preprocesses" in self.datamodule_conf:
            for key, value in self.datamodule_conf["single_preprocesses"].items():
                if key in key_selects:
                    precrocesses[key] = init_preporcesses(value)
        for key, value in self.datamodule_conf['datasets'].items():
            if key in key_selects:
                self.datasets[key] = AMINO_ConcatDataset(datasets) \
                    if (datasets := init_list_object(value)) is not None else None
                if self.datasets[key] is not None:
                    logging.info(f"the {key} dataset len: {len(self.datasets[key])}")
                    self.datasets[key].set_preprocesses(
                        precrocesses.get(key, None)
                    )
                self.collect_fns[key] = AMINOPadCollate(
                    **self.datamodule_conf['collect_fns'][key]
                )

    def train_dataloader(self):
        return self.x_dataloader("train")

    def val_dataloader(self):
        return self.x_dataloader("val")

    def test_dataloader(self):
        return self.x_dataloader("test")

    def x_dataloader(self, datasetname):
        if self.datasets[datasetname] is not None:
            logging.info(
                f"there are {len(self.datasets[datasetname])} items in {datasetname} dataset"
            )
            dataloader = tdata.DataLoader(
                self.datasets[datasetname],
                **self.datamodule_conf['dataloaders'][datasetname],
                collate_fn=self.collect_fns[datasetname],
            )
            logging.info(
                f"there are {len(dataloader)} batches in {datasetname} dataloader"
            )
            return dataloader
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

    def get_dataset(self, datasetname):        
        return self.datasets[datasetname] if datasetname in self.datasets else None
    