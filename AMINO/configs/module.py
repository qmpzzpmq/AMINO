from abc import ABC
from typing import Union

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from AMINO.configs.common import AMINO_CONF

@type_checked_constructor()
@dataclass
class LOSSES(ABC):
    # net: Dict[str, AMINO_CONF] = field(default_factory=Dict)
    # weight: Dict[str, float] = field(default_factory=Dict)
    net: dict = field(default_factory=dict)
    weight: dict = field(default_factory=dict)

@dataclass
class OPTIM(AMINO_CONF, ABC):
    contiguous_params: bool = False

@type_checked_constructor()
@dataclass
class MODULE_CONF(ABC):
    # losses: LOSSES = LOSSES(
    #     net={"autoencoder": AMINO_CONF(select="torch.nn:MSELoss", conf={"reduction": "none"})},
    #     weight={"autoencoder": 1.0},
    # )
    losses: Union[LOSSES, None] = None
    optim: OPTIM = OPTIM(
        select="torch.optim:Adam", 
        contiguous_params=False,
        conf={"lr": 0.01}
    )
    # scheduler: AMINO_CONF = AMINO_CONF(
    #     select="torch.optim.lr_scheduler:StepLR",
    #     conf={"step_size": 5, "gamma": 0.1}
    # )
    scheduler: AMINO_CONF = AMINO_CONF(
        select="AMINO.modules.scheduler:WarmupLR",
        conf={"warmup_steps": 2000}
    )
    net: AMINO_CONF = AMINO_CONF(
        select="AMINO.modules.nets.autoencoder:simple_autoencoder",
        conf={},
    )