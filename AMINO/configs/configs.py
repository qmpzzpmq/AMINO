from abc import ABC
from typing import Any, Dict, Optional, Union, List

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
    # it will be fixed in https://github.com/omry/omegaconf/issues/144
    # trainer: TRAINER =  field(default_factory=lambda: TRAINER(
    trainer: dict =  field(default_factory=lambda: dict(
        accelerator = None,
        accumulate_grad_batches = None,
        amp_backend = 'native',
        amp_level = None,
        auto_lr_find = False,
        auto_scale_batch_size = False,
        auto_select_gpus = False,
        benchmark = False,
        # callbacks = None,
        enable_checkpointing = True,
        check_val_every_n_epoch = 1,
        default_root_dir = None,
        detect_anomaly = False,
        deterministic = False,
        devices = None,
        fast_dev_run = False,
        gpus = None,
        gradient_clip_val = None,
        gradient_clip_algorithm = None,
        limit_train_batches = 1.0,
        limit_val_batches = 1.0,
        limit_test_batches = 1.0,
        limit_predict_batches = 1.0,
        # limit_val_batches: (Union[int, float])
        # limit_test_batches: (Union[int, float])
        # limit_predict_batches: (Union[int, float])
        # logger = True,
        log_every_n_steps = 50,
        enable_progress_bar = True,
        profiler = None,
        overfit_batches = 0.0,
        # plugins: # not need here
        precision = 32,
        max_epochs = None,
        min_epochs =  None,
        max_steps = -1,
        min_steps = None,
        max_time = None,
        num_nodes = 1,
        num_processes = 1,
        num_sanity_val_steps = 2,
        reload_dataloaders_every_n_epochs = 0,
        replace_sampler_ddp = True,
        strategy = None,
        sync_batchnorm = False,
        terminate_on_nan = None,
        tpu_cores = None,
        ipus = None,
        track_grad_norm = -1,
        val_check_interval = 1.0,
        weights_save_path = None,
        move_metrics_to_cpu = False,
        multiple_trainloader_mode = 'max_size_cycle',
    ))
    variables: Any = None
    pipeline_size: Any = None
    feature_statistics: FEATURE_STATISTICS = field(default_factory=FEATURE_STATISTICS)
    checkpoint: Optional[Union[None, str]] = None 

def register_OmegaConf_resolvers():
    OmegaConf.register_new_resolver("nfft2fea_dim", lambda x: int(x / 2 + 1))
    OmegaConf.register_new_resolver("product", lambda x, y: x * y)
