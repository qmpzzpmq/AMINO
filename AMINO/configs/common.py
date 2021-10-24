from abc import ABC
from typing import Any, Optional, Dict, List

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

@type_checked_constructor()
@dataclass
class AMINO_CONF(ABC):
    select: str = ""
    conf: Any = field(default_factory=dict)