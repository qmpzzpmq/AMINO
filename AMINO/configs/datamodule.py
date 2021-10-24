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
    val: DATALOADER = field(default_factory=DATALOADER)
    test: Union[DATALOADER, None] = field(default_factory=None)

@type_checked_constructor()
@dataclass
class DATASETS(ABC):
    train: AMINO_CONF = AMINO_CONF(
        select="AMINO.datamodule.datasets:TOYADMOS2_DATASET",
    )
    val: Union[AMINO_CONF, None] = field(default_factory=None)
    test: Union[AMINO_CONF, None] = field(default_factory=None)

@type_checked_constructor()
@dataclass
class TRANSFORMS(ABC):
    train: Union[List[AMINO_CONF], None] = field(default_factory=None)
    val: Union[List[AMINO_CONF], None] = field(default_factory=None)
    test: Union[List[AMINO_CONF], None] = field(default_factory=None)

@type_checked_constructor()
@dataclass
class DATAMODULE(ABC):
    datasets: DATASETS = field(default_factory=DATASETS)
    dataloaders: DATALOADERS= field(default_factory=DATALOADERS)
    single_preprocesses: TRANSFORMS = TRANSFORMS(
        train=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:AUDIO_GENERAL",
                conf={"fs":16000, "mono_channel": 'mean'},
            ),
        ],
        val=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:AUDIO_GENERAL",
                conf={"fs":16000, "mono_channel": 'mean'}
            ),
        ],
        test=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:AUDIO_GENERAL",
                conf={"fs":16000, "mono_channel": 'mean'},
            ),
        ],
    )
    after_transform: TRANSFORMS = TRANSFORMS(
        train=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:FFT",
                conf={"n_fft": 512},
            ),
        ],
        val=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:FFT",
                conf={"n_fft": 512},
            )
        ],
        test=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:FFT",
                conf={"n_fft": 512},
            )
        ],
    )
