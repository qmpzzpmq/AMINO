from abc import ABC

from dataclasses import dataclass, field
from undictify import type_checked_constructor

@type_checked_constructor()
@dataclass
class PROGRESSBAR_CONF(ABC):
    refresh_rate: int = 1
    process_position: int = 0

@type_checked_constructor()
@dataclass
class MODELCHECKPOINT_CONF(ABC):
    filename: str ='epoch{epoch}-val_loss{val_normal_loss:.3f}' 
    monitor: str = 'val_normal_loss'
    save_last: bool = True
    save_top_k: int = 5
    dirpath: str = 'checkpoint'

@type_checked_constructor()
@dataclass
class EARLYSTOPPING_CONF(ABC):
    monitor: str = 'val_normal_loss'
    mode: str = 'max'
    min_delta: float = 0.001
    patience: int = 5

@type_checked_constructor()
@dataclass
class GPU_STATS_CONF(ABC):
    memory_utilization: bool = True
    gpu_utilization: bool = True

@type_checked_constructor()
@dataclass
class LR_MONITOR_CONF(ABC):
    logging_interval: str = 'epoch'

@type_checked_constructor()
@dataclass
class CALLBACKS(ABC):
    progressbar: bool = True
    progressbar_conf: PROGRESSBAR_CONF = field(default_factory=PROGRESSBAR_CONF)
    modelcheckpoint: bool = True
    modelcheckpoint_conf: MODELCHECKPOINT_CONF = field(default_factory=MODELCHECKPOINT_CONF)
    earlystopping: bool = True
    earlystopping_conf: EARLYSTOPPING_CONF = field(default_factory=EARLYSTOPPING_CONF)
    gpu_stats: bool = False
    gpu_stats_conf: GPU_STATS_CONF = field(default_factory=GPU_STATS_CONF)
    lr_monitor: bool = True
    lr_monitor_conf: LR_MONITOR_CONF = field(default_factory=LR_MONITOR_CONF)