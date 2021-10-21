import torch.utils.data as tdata
import pytorch_lightning as pl

from AMINO.datamodule.datasets import init_dataset
from AMINO.datamodule.preprocess import init_preporcesses
from AMINO.datamodule.utils import MulPadCollate, SinglePadCollate

class AMINODataModule(pl.LightningDataModule):
    def __init__(self, datamodule_conf):
        super().__init__()
        self.datamodule_conf = datamodule_conf
        self.save_hyperparameters()
        self.datasets = dict()
        self.collect_fns = dict()
        self.transform = {'after': dict()}
        for dataset in ['train', 'val', 'test']:
            self.transform['after'][dataset] = self.preprocess_get(
                'after_transform', dataset
            )

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
                    self.datasets[key] = init_dataset(value)
                    self.datasets[key].set_preprocesses(
                        precrocesses.get(key, None)
                    )
                    self.collect_fns[key] = SinglePadCollate(-1)

    def train_dataloader(self):
        if 'train' in self.datasets:
            return tdata.DataLoader(
                self.datasets['train'],
                **self.datamodule_conf['dataloaders']['train'],
                collate_fn=self.collect_fns['train']
            )
        else:
            return None

    def val_dataloader(self):
        if 'val' in self.datasets:
            return tdata.DataLoader(
                self.datasets['val'],
                **self.datamodule_conf['dataloaders']['val'],
                collate_fn=self.collect_fns['val']
            )
        else:
            return None

    def test_dataloader(self):
        if 'test' in self.datasets:
            return tdata.DataLoader(
                self.datasets['test'],
                **self.datamodule_conf['dataloaders']['test'],
                collate_fn=self.collect_fns['test']
            )
        else:
            return None

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if type(self.trainer.accelerator) == pl.accelerators.cpu.CPUAccelerator:
            batch = self.on_after_batch_transfer(batch, dataloader_idx)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training \
                and self.transform['after'].get('train', None) is not None:
            return self.transform['after']['train'](batch)
        elif (self.trainer.validating or self.trainer.sanity_checking) \
                and self.transform['after'].get('val', None) is not None:
            return self.transform['after']['val'](batch)
        elif self.trainer.testing \
                and self.transform['after'].get('testing', None) is not None:
            return self.transform['after']['testing'](batch)
        return batch
    
    def preprocess_get(self, position, key):
        if position in self.datamodule_conf and \
                key in self.datamodule_conf[position]:
            return init_preporcesses(
                self.datamodule_conf[position][key]
            )
        else:
            return None


    def tesrdown(self):
        super().teardown()

def init_datamodule(datamodule_conf):
    return AMINODataModule(datamodule_conf)
