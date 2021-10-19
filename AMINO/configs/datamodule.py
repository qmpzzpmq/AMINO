from abc import ABC
from typing import Any, List, Union

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from AMINO.configs.common import AMINO_CONF

@type_checked_constructor()
@dataclass
class DATALOADER(ABC):
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0

@type_checked_constructor()
@dataclass
class DATALOADERS(ABC):
    train: DATALOADER = field(default_factory=DATALOADER)
    dev: Union[DATALOADER, None] = field(default_factory=None)
    test: Union[DATALOADER, None]  = field(default_factory=None)
@type_checked_constructor()
@dataclass
class DATASET_CONF(ABC):
    path: str 
    mono: str = "mean" #could be "mean" "0" "1"
    fs: int = 16000

@type_checked_constructor()
@dataclass
class DATASETS(ABC):
    train: AMINO_CONF = AMINO_CONF(select="TOYADMOS2_DATASET")
    dev: Union[AMINO_CONF, None] = field(default_factory=None)
    test: Union[AMINO_CONF, None] = field(default_factory=None)

@type_checked_constructor()
@dataclass
class PREPORCESSES(ABC):
    train: List = field(default_factory=
        lambda: [
            AMINO_CONF(select="AUDIO_GENERAL", conf={"fs":16000, "mono_channel": 'mean'}),
            AMINO_CONF(select="FFT"),
        ]
    )
    dev: AMINO_CONF = field(default_factory=
        lambda: [
            AMINO_CONF(select="AUDIO_GENERAL", conf={"fs":16000, "mono_channel": 'mean'}),
            AMINO_CONF(select="FFT"),
        ]
    )
    test: AMINO_CONF = field(default_factory=
        lambda: [
            AMINO_CONF(select="AUDIO_GENERAL", conf={"fs":16000, "mono_channel": 'mean'}),
            AMINO_CONF(select="FFT"),
        ]
    )

@type_checked_constructor()
@dataclass
class DATAMODULE(ABC):
    preprocess: PREPORCESSES = field(default_factory=PREPORCESSES)
    datasets: DATASETS = field(default_factory=DATASETS)
    dataloaders: DATALOADERS= field(default_factory=DATALOADERS)