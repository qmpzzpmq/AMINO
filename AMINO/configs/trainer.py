from abc import ABC

from dataclasses import dataclass, MISSING
from undictify import type_checked_constructor

@type_checked_constructor()
@dataclass
class TRAINER(ABC):
    accelerator: str = 'ddp'
    accumulate_grad_batches: int = 1
    amp_backend: str = 'native' # 'native' 'apex'
    max_epochs: int = 100
    min_epochs: int = 5
    amp_level: str = 'O0'
    auto_lr_find: bool = False
    auto_scale_batch_size: bool = False
    auto_select_gpus: bool = True
    benchmark: bool = False
    fast_dev_run: bool = False
    flush_logs_every_n_steps: int = 100
    gpus: str = ""
    gradient_clip_val: int = 50
    gradient_clip_algorithm: str = 'norm'
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    limit_test_batches = 1.0
    limit_predict_batches = 1.0
    log_every_n_steps: int = 5
    precision: int = 32
    replace_sampler_ddp: bool = True
    resume_from_checkpoint: str = ""