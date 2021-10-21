from abc import ABC
from typing import Any

from omegaconf import OmegaConf
from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from AMINO.configs.datamodule import DATAMODULE
from AMINO.configs.module import MODULE_CONF
from AMINO.configs.callbacks import CALLBACKS
from AMINO.configs.loggers import LOGGERS
from AMINO.configs.trainer import TRAINER
from AMINO.configs.common import HYDRA, AMINO_CONF


@type_checked_constructor()
@dataclass
class EXP_BASE(ABC):
  exp: str = 'exp'
  tensorboard: str = 'tensorboard'
  wandb: str = 'wandb'
  neptune: str = 'neptune'
  seed: int = 777

@type_checked_constructor()
@dataclass
class LOGGING(ABC):
    level: str = "DEBUG"

@type_checked_constructor()
@dataclass
class TRAIN_CONFIG():
    datamodule: DATAMODULE = field(default_factory=DATAMODULE)
    module: AMINO_CONF = AMINO_CONF(
        select='AMINO_AUTOENCODER', conf=OmegaConf.structured(MODULE_CONF))
    expbase: EXP_BASE = field(default_factory=EXP_BASE)
    callbacks: CALLBACKS = field(default_factory=CALLBACKS)
    loggers: LOGGERS = field(default_factory=LOGGERS)
    logging: LOGGING = field(default_factory=LOGGING)
    trainer: TRAINER = field(default_factory=TRAINER)
    temp: Any = None
    expname: str = "baseline"
    hydra: HYDRA = field(default_factory=HYDRA)

def register_OmegaConf_resolvers():
    OmegaConf.register_new_resolver("nfft2fea_dim", lambda x: int(x / 2 + 1))

# unavailable
def cfg_check(cfg):
    if 'feature_dim' in cfg['module']['conf']['net_conf']:
        feature_dim = cfg['module']['conf']['net_conf']['feature_dim']
        for dataloader_name in ['train', 'val', 'test']:
            if dataloader_name in cfg['datamodule']['dataloaders']:
                dataloader = cfg['datamodule']['dataloaders'][dataloader_name]
                for transform in dataloader['after_transform']:
                    if transform['select'] == "FFT":    
                        nfft = transform['conf']['nfft']
                        assert feature_dim == nfft / 2 + 1, \
                            "please check the feature_dim {feature_dim} with nfft {nfft}, unmatch"