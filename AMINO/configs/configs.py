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
    level: str = "INFO"

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
