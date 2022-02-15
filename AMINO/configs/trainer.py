from abc import ABC
from typing import Any, Union, List, Optional, Dict, Iterable
from datetime import timedelta

from attrs import asdict, define, make_class, Factory
from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler

@type_checked_constructor()
@dataclass
class _TRAINER(ABC):
    accelerator: (Union[str, None]) = None
    accumulate_grad_batches: int = 1
    amp_backend: str = 'native' # 'native' 'apex'
    max_epochs: int = 100
    min_epochs: int = 5
    amp_level: Union[str, None] = None
    auto_lr_find: bool = False
    # auto_scale_batch_size: Union[str, bool, None] = None
    auto_scale_batch_size: Any = False # Temporare method
    auto_select_gpus: bool = False
    benchmark: bool = False
    fast_dev_run: bool = False
    flush_logs_every_n_steps: int = 100
    # gpus: Union[int, str, List[int], None] = field(default_factory=None)
    gpus: Any = None # Temporare method
    gradient_clip_val: int = 50
    gradient_clip_algorithm: str = 'norm'
    limit_train_batches: Union[int, float] = field(default=1.0)
    # limit_val_batches: (Union[int, float]) = 1.0
    # limit_test_batches: (Union[int, float]) = 1.0
    # limit_predict_batches: (Union[int, float]) = 1.0
    log_every_n_steps: int = 5
    precision: int = 32
    replace_sampler_ddp: bool = True
    profiler: Union[str, None] = None
    strategy: Union[str, None] = "ddp" # should be pytorch_lightning > 1.5

# Union types are not supported (except Optional)
# https://hydra.cc/docs/next/tutorials/structured_config/intro/#structured-configs-limitations

@type_checked_constructor()
@dataclass
class TRAINER(ABC):
    accelerator: Union[str, None]
    # accumulate_grad_batches: Union[int, Dict[int, int], None]
    accumulate_grad_batches: Union[int, None]
    amp_backend: str # 'native' 'apex'
    amp_level: Optional[str]
    auto_lr_find: Union[bool, str]
    auto_scale_batch_size: Union[str, bool]
    auto_select_gpus: bool
    benchmark: bool
    callbacks: Union[List[Callback], Callback, None]
    enable_checkpointing: bool
    check_val_every_n_epoch: int
    default_root_dir: Optional[str]
    detect_anomaly: bool
    deterministic: bool
    devices: Union[int, str, List[int], None]
    fast_dev_run: bool
    gpus: Union[int, str, List[int], None]
    gradient_clip_val: Union[int, float, None]
    gradient_clip_algorithm: Optional[str]
    limit_train_batches: Union[int, float]
    # limit_val_batches: (Union[int, float])
    # limit_test_batches: (Union[int, float])
    # limit_predict_batches: (Union[int, float])
    # logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    logger: bool
    log_every_n_steps: int
    enable_progress_bar: bool
    # profiler: Union[BaseProfiler, str, None]
    profiler: Union[str, None]
    overfit_batches: Union[int, float]
    # plugins: # not need here
    precision: Union[int, str]
    max_epochs: Optional[int]
    min_epochs: Optional[int]
    max_step: int
    min_step: Optional[int]
    # max_time: Union[str, timedelta, Dict[str, int], None]
    max_time: Union[str, Dict[str, int], None]
    num_nodes: int
    num_processes: int
    num_sanity_val_steps: int
    reload_dataloaders_every_n_epochs: int
    replace_sampler_ddp: bool
    strategy: Union[str, None]
    sync_batchnorm: bool
    terminate_on_nan: Optional[bool]
    tpu_cores: Union[int, str, List[int], None]
    ipus: Optional[int]
    track_grad_norm: Union[int, float, str]
    val_check_interval: Union[int, float]
    weights_save_path: Optional[str]
    move_metrics_to_cpu: bool
    multiple_trainloader_mode: str

@define
class TRAINER_(ABC):
    accelerator: Union[str, None]
    # accumulate_grad_batches: Union[int, Dict[int, int], None]
    accumulate_grad_batches: Union[int, None]
    amp_backend: str # 'native' 'apex'
    amp_level: Optional[str]
    auto_lr_find: Union[bool, str]
    auto_scale_batch_size: Union[str, bool]
    auto_select_gpus: bool
    benchmark: bool
    callbacks: Union[List[Callback], Callback, None]
    enable_checkpointing: bool
    check_val_every_n_epoch: int
    default_root_dir: Optional[str]
    detect_anomaly: bool
    deterministic: bool
    devices: Union[int, str, List[int], None]
    fast_dev_run: bool
    gpus: Union[int, str, List[int], None]
    gradient_clip_val: Union[int, float, None]
    gradient_clip_algorithm: Optional[str]
    limit_train_batches: Union[int, float]
    # limit_val_batches: (Union[int, float])
    # limit_test_batches: (Union[int, float])
    # limit_predict_batches: (Union[int, float])
    # logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    logger: bool
    log_every_n_steps: int
    enable_progress_bar: bool
    # profiler: Union[BaseProfiler, str, None]
    profiler: Union[str, None]
    overfit_batches: Union[int, float]
    # plugins: # not need here
    precision: Union[int, str]
    max_epochs: Optional[int]
    min_epochs: Optional[int]
    max_step: int
    min_step: Optional[int]
    # max_time: Union[str, timedelta, Dict[str, int], None]
    max_time: Union[str, Dict[str, int], None]
    num_nodes: int
    num_processes: int
    num_sanity_val_steps: int
    reload_dataloaders_every_n_epochs: int
    replace_sampler_ddp: bool
    strategy: Union[str, None]
    sync_batchnorm: bool
    terminate_on_nan: Optional[bool]
    tpu_cores: Union[int, str, List[int], None]
    ipus: Optional[int]
    track_grad_norm: Union[int, float, str]
    val_check_interval: Union[int, float]
    weights_save_path: Optional[str]
    move_metrics_to_cpu: bool
    multiple_trainloader_mode: str