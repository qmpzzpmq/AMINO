from abc import ABC
from typing import Any, Dict, Union, List

from omegaconf import OmegaConf
from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from AMINO.configs.datamodule import DATAMODULE
from AMINO.configs.module import MODULE_CONF
from AMINO.configs.trainer import TRAINER
from AMINO.configs.common import AMINO_CONF


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
class FEATURE_STATISTICS(ABC):
    set: str = "val"

@type_checked_constructor()
@dataclass
class TRAIN_CONFIG():
    datamodule: DATAMODULE = field(default_factory=DATAMODULE)
    module: AMINO_CONF = AMINO_CONF(
        select='AMINO.modules.autoencoder:AMINO_AUTOENCODER',
        conf=OmegaConf.structured(MODULE_CONF),
    )
    expbase: EXP_BASE = field(default_factory=EXP_BASE)
    callbacks: Union[List[AMINO_CONF], None] = field(default_factory=lambda: [
        AMINO_CONF(
            select="pytorch_lightning.callbacks.progress.tqdm_progress:TQDMProgressBar",
            conf={
                "refresh_rate": 1,
                "process_position": 0,
            },
        ),
        AMINO_CONF(
            select="pytorch_lightning.callbacks.model_checkpoint:ModelCheckpoint",
            conf={
                "filename": 'epoch{epoch}-val_normal_loss{val_normal_loss:.3f}',
                "monitor":  'val_normal_loss_epoch',
                "save_last": True,
                "save_top_k": 5,
                "dirpath": 'checkpoint',
            },
        ),
        AMINO_CONF(
            select="pytorch_lightning.callbacks.early_stopping:EarlyStopping",
            conf={
                "monitor": 'val_normal_loss_epoch',
                "mode": 'min',
                "min_delta": 1e-6,
                "patience": 30,
            },
        ),
        AMINO_CONF(
            select="pytorch_lightning.callbacks:DeviceStatsMonitor",
        ),
        AMINO_CONF(
            select="pytorch_lightning.callbacks.lr_monitor:LearningRateMonitor",
            conf={
                "logging_interval": 'epoch',
            },
        ),
        AMINO_CONF(
            select="pytorch_lightning.callbacks:ModelSummary",
            conf={
                "max_depth": 3,
            },
        ),
    ])
    loggers: Union[List[AMINO_CONF], None] = field(default_factory=lambda: [
        AMINO_CONF(
            select="pytorch_lightning.loggers:TensorBoardLogger",
            conf={
                "save_dir": '${expbase.tensorboard}/${hydra:job.name}',
            },
        ),
        AMINO_CONF(
            select="pytorch_lightning.loggers.wandb:WandbLogger",
            conf={
                "name": '${hydra:job.name}',
                "save_dir": None,
                "project": "AMINO",
                "log_model": False,
            },
        ),
    ])
    logging: LOGGING = field(default_factory=LOGGING)
    trainer: TRAINER = field(default_factory=TRAINER)
    variables: Any = None
    hydra: Any = None
    feature_statistics: FEATURE_STATISTICS = field(default_factory=FEATURE_STATISTICS)

def register_OmegaConf_resolvers():
    OmegaConf.register_new_resolver("nfft2fea_dim", lambda x: int(x / 2 + 1))
    OmegaConf.register_new_resolver("product", lambda x, y: x * y)
