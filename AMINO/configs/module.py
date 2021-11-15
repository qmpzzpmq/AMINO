from abc import ABC
from typing import Any

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from AMINO.configs.common import AMINO_CONF

@dataclass
class OPTIM(AMINO_CONF, ABC):
    contiguous_params: bool = False

@type_checked_constructor()
@dataclass
class MODULE_CONF(ABC):
    loss_conf: AMINO_CONF = AMINO_CONF(select="torch.nn:MSELoss", conf={"reduction": "none"})
    optim_conf: OPTIM = OPTIM(
        select="torch.optim:Adam", 
        contiguous_params=False,
        conf={"lr": 0.01}
    )
    scheduler_conf: AMINO_CONF = AMINO_CONF(
        select="torch.optim.lr_scheduler:StepLR",
        conf={"step_size": 5, "gamma": 0.1}
    )
    net_conf: AMINO_CONF = AMINO_CONF(
        select="AMINO.modules.nets.autoencoder:simple_autoencoder",
        conf={},
    )