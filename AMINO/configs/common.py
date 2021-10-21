from abc import ABC
from typing import Any, Optional

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

@type_checked_constructor()
@dataclass
class AMINO_CONF(ABC):
    select: str = ""
    conf: Any = field(default_factory=dict)

@type_checked_constructor()
@dataclass
class RUN(ABC):
    dir: str = field(default_factory=str)

@type_checked_constructor()
@dataclass
class HYDRA(ABC):
    name: str = field(default_factory=str)
    output_subdir = None
    run: RUN = field(default_factory=RUN)
    hydra_logging = None
    log_logging = None