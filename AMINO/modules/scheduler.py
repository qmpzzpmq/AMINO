import torch

from AMINO.utils.dynamic_import import dynamic_import

def init_scheduler(optim, scheduler_conf):
    scheduler_class = dynamic_import(scheduler_conf['select'])
    scheduler = scheduler_class(optim, **scheduler_conf['conf'])
    return scheduler 