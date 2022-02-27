import logging

import torch
import torch.utils.data as tdata
import torch.nn as nn
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
        self.transform = {'batch_after': dict(), "batch_before": dict}
        self.transform2device = {'batch_after': dict()}
        # for dataset_name in ['train', 'val', 'test']:
        #     self.transform['batch_after'][dataset_name] = self.preprocess_get(
        #         'batch_after', dataset_name
        #     )
        #     self.transform2device['batch_after'][dataset_name] = False
        #     
        for dataset_name in ['train', 'val', 'test']:
            self.dataset_init(dataset_name)

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
        # download
        # tokenize
        # etc…
        for dataset_name, dataset in self.datasets.items():
            if dataset:
                for subdataset in self.datasets[dataset_name]:
                    subdataset.prepare_data()
        super().prepare_data()

    def setup(self, stage=None):
        # There are also data operations you might want to perform on every GPU. Use setup to do things like:
        # count number of classes
        # build vocabulary
        # perform train/val/test splits
        # apply transforms (defined explicitly in your datamodule)
        # etc…
        for dataset_name in self.datasets.keys():
            if self.datasets[dataset_name]:
                for subdataset in self.datasets[dataset_name]:
                    subdataset.setup()
                concat_dataset = AMINO_ConcatDataset(
                    self.datasets[dataset_name]
                )
                if concat_dataset is not None:
                    self.datasets[dataset_name] = concat_dataset
                    logging.info(
                        f"the {dataset_name} dataset len: {len(self.datasets[dataset_name])}"
                    )
                    self.datasets[dataset_name].set_preprocesses(
                        self.preprocess_get('item', dataset_name)
                    )
                    self.collect_fns[dataset_name] = AMINOPadCollate(
                        **self.datamodule_conf['collect_fns'][dataset_name]
                    )
                    self.transform['batch_after'][dataset_name] = self.preprocess_get(
                        'batch_after', dataset_name
                    )
                    self.transform2device[dataset_name] = False
        super().setup()
    
    def get_num_classes(self):
        if hasattr(self, "num_classes"):
            return self.num_classes
        try:
            num_classes = list()
            for dataset in self.datasets.values():
                if dataset is not None:
                    num_classes += [
                        subset.get_num_classes() for subset in dataset
                    ]
            assert len(num_classes) > 0, "all dataset empty"
            num_classes = torch.tensor(num_classes)
            logging.debug(f"num_classes: {num_classes}")
            assert num_classes.max() == num_classes.min(), \
                f"num_classes in each data is not same"
            num_classes = num_classes.max().item()
        except Exception as e:
            logging.warning(
                f"Something {e} wrong with get_num_classes, return None"
            )
            num_classes = None
        self.num_classes = num_classes
        return self.num_classes

    def dataset_init(self, key_selects):
        precrocesses = dict()
        if "single_preprocesses" in self.datamodule_conf:
            for key, value in self.datamodule_conf["single_preprocesses"].items():
                if key in key_selects:
                    precrocesses[key] = init_preporcesses(value)
        for key, value in self.datamodule_conf['datasets'].items():
            if key in key_selects:
                self.datasets[key] = datasets \
                    if (datasets := init_list_object(value)) is not None else None

    def train_dataloader(self):
        return self.x_dataloader("train")

    def val_dataloader(self):
        return self.x_dataloader("val")

    def test_dataloader(self):
        return self.x_dataloader("test")

    def set_replace_sampler_ddp(self, replace_sampler_ddp):
        self.replace_sampler_ddp = replace_sampler_ddp

    def x_dataloader(self, datasetname):
        if self.datasets[datasetname] is not None:
            dataset = self.datasets[datasetname]
            logging.info(
                f"there are {len(dataset)} items in {datasetname} dataset"
            )
            dataloader_conf = self.datamodule_conf['dataloaders'][datasetname]
            if not self.replace_sampler_ddp:
                shuffle = True if datasetname == "train" else False
                sampler = tdata.distributed.DistributedSampler(
                    dataset, shuffle = shuffle)
                logging.info(f"sampler len: {len(sampler)}")
                dataloader_conf["shuffle"] = False
            else:
                sampler = None
            logging.info(f"{datasetname} dataloader conf: {dataloader_conf}")
            dataloader = tdata.DataLoader(
                dataset,
                **dataloader_conf,
                collate_fn=self.collect_fns[datasetname],
                sampler=sampler,
            )
            logging.info(
                f"there are {len(dataloader)} batches in {datasetname} dataloader"
            )
            return dataloader
        else:
            return None

    # this bug fix: when pytorch_lightning > 1.5
    # if pytorch_lightning < 1.5 using this code
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     if type(self.trainer.accelerator) == pl.accelerators.cpu.CPUAccelerator:
    #         print("0")
    #         return self.on_after_batch_transfer(batch, dataloader_idx)
    #     print(f"1. accelerator: {type(self.trainer.accelerator)}")
    #     return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training :
            return self.batch_transform('batch_after', 'train', batch)
        elif (self.trainer.validating or self.trainer.sanity_checking):
            return self.batch_transform('batch_after', 'val', batch)
        elif self.trainer.testing:
            return self.batch_transform('batch_after', 'test', batch)
        elif self.trainer.predicting :
            return self.batch_transform('batch_after', 'predict', batch)
        return batch

    def batch_transform(self, position, key, batch):
        if not self.transform2device[key]:
            self.transform[position][key] = \
                self.transform[position][key].to(
                    batch['feature']['data'].device
                )
            self.transform2device[key] = True
        batch = self.transform[position][key](batch)
        return batch

    def preprocess_get(self, position, key):
        if position in self.datamodule_conf["transform"] and \
                key in self.datamodule_conf["transform"][position]:
            preprocesses = init_list_object(
                self.datamodule_conf["transform"][position][key]
            )
            return torch.nn.Sequential(*preprocesses) \
                if preprocesses is not None else nn.Identity()
        else:
            return nn.Identity()

    def tesrdown(self):
        super().teardown()

    def get_dataset(self, datasetname):        
        return self.datasets[datasetname] if datasetname in self.datasets else None
    