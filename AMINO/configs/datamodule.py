from abc import ABC
from typing import Any, List, Union

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

from AMINO.configs.common import AMINO_CONF

@type_checked_constructor()
@dataclass
class DATALOADER(ABC):
    batch_size: int = 1
    shuffle: bool = True
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
    train: Union[List[AMINO_CONF], None] = field(default_factory=None)
    val: Union[List[AMINO_CONF], None] = field(default_factory=None)
    test: Union[List[AMINO_CONF], None] = field(default_factory=None)

@type_checked_constructor()
@dataclass
class COLLECT_FN(ABC):
    pad_choices: List[str] = field(default_factory=None)

@type_checked_constructor()
@dataclass
class COLLECT_FNS(ABC):
    train: COLLECT_FN = COLLECT_FN(
        pad_choices=["pdb", "unpad"],
    )
    val: COLLECT_FN = COLLECT_FN(
        pad_choices=["pdb", "unpad"],
    )
    test: COLLECT_FN = COLLECT_FN(
        pad_choices=["pdb", "unpad"],
    )

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
    collect_fns: COLLECT_FNS = field(default_factory=COLLECT_FNS)
    single_preprocesses: TRANSFORMS = TRANSFORMS(
        train= None,
        val= None,
        test= None,
    )
    after_transform: TRANSFORMS = TRANSFORMS(
        train=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:MelSpectrogram",
                conf={
                    "n_fft": 512,
                    "n_mels": 128,
                },
            ),
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:SpecAug",
                conf={
                    "frequency_mask": {
                        "F": 30,
                        "num_mask": 2,
                    },
                    "time_mask": {
                        "T": 40,
                        "num_mask": 2,
                    },
                    # "time_stretch": {
                    #     "floor": 0.9,
                    #     "ceil": 1.1
                    # },
                }
            ),
        ],
        val=[
            AMINO_CONF(
                select="AMINO.datamodule.preprocess:Spectrogram",
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
