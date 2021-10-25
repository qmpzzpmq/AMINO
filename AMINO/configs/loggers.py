from abc import ABC
from typing import Any, List, Union

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

@type_checked_constructor()
@dataclass
class TENSORBOARD_CONF(ABC):
    save_dir: str = '${expbase.tensorboard}/${hydra:job.name}'

@type_checked_constructor()
@dataclass
class WANDB_CONF(ABC):
    name: str = '${hydra:job.name}'
    save_dir: str = '${expbase.wandb}'
    project: str = "AMINO"
    log_model: bool = False

@type_checked_constructor()
@dataclass
class NEPTUNE_CONF(ABC):
    project_name: str = '${expbase.neptune}'
    experiment_name: str = "${hydra:job.name}"
    api_token: str = "ANONYMOUS"

@type_checked_constructor()
@dataclass
class LOGGERS(ABC):
    tensorboard: bool = False
    tensorboard_conf: TENSORBOARD_CONF = field(default_factory=TENSORBOARD_CONF)
    wandb: bool = False
    wandb_conf: WANDB_CONF = field(default_factory=WANDB_CONF)
    neptune: bool = False
    neptune_conf: NEPTUNE_CONF = field(default_factory=NEPTUNE_CONF)