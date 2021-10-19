from abc import ABC
from typing import Any

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from AMINO.configs.common import AMINO_CONF

@type_checked_constructor()
@dataclass
class FFT(ABC):
    n_fft: 512
    hop_length = 400

@type_checked_constructor()
@dataclass
class FRONTEND(ABC):
    fft: FFT = field(default_factory=FFT)

@type_checked_constructor()
@dataclass
class OPTIM(ABC):
    select: str = ""
    contiguous_params: bool = False
    conf: Any = field(default_factory=dict)

@type_checked_constructor()
@dataclass
class MODULE_CONF(ABC):
    loss: AMINO_CONF = AMINO_CONF(select="torch.nn.MSELoss", conf={"reduction": "sum"})
    optim: OPTIM = OPTIM(
        select="torch.optim.Adam", contiguous_params=True, conf={"lr": "0.001"}
    )
    scheduler: AMINO_CONF = AMINO_CONF(
        select="lambdalr", conf={"last_epoch": -1, "lr_lambda":'lambda epoch: 0.95 ** epoch'}
    )
    net: AMINO_CONF = AMINO_CONF(select="simple_autoencoder")