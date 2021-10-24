from abc import ABC
from typing import Any, Optional, Dict, List

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

@type_checked_constructor()
@dataclass
class AMINO_CONF(ABC):
    select: str = ""
    conf: Any = field(default_factory=dict)

@type_checked_constructor()
@dataclass
class Job:
    # Job name, populated automatically unless specified by the user (in config or cli)
    name: str = MISSING
    # Concatenation of job overrides that can be used as a part
    # of the directory name.
    # This can be configured in hydra.job.config.override_dirname
    override_dirname: str = MISSING
    # Job ID in underlying scheduling system
    id: str = MISSING
    # Job number if job is a part of a sweep
    num: int = MISSING
    # The config name used by the job
    config_name: Optional[str] = MISSING
    # Environment variables to set remotely
    env_set: Dict[str, str] = field(default_factory=dict)
    # Environment variables to copy from the launching machine
    env_copy: List[str] = field(default_factory=list)
    # Job config
    @dataclass
    class JobConfig:
        @dataclass
        # configuration for the ${hydra.job.override_dirname} runtime variable
        class OverrideDirname:
            kv_sep: str = "="
            item_sep: str = ","
            exclude_keys: List[str] = field(default_factory=list)

        override_dirname: OverrideDirname = OverrideDirname()

    config: JobConfig = JobConfig()

@type_checked_constructor()
@dataclass
class RUN(ABC):
    dir: str = 'exp/${now:%Y-%m-%d}/${now:%H-%M-%S}'

@type_checked_constructor()
@dataclass
class HYDRA(ABC):
    name: str = field(default_factory=str)
    output_subdir = None
    run: RUN = field(default_factory=RUN)
    hydra_logging = None
    log_logging = None
    job: Job = field(default_factory=Job)