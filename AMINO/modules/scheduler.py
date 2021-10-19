import torch

def init_scheduler(optim, scheduler_conf):
    scheduler_select = scheduler_conf.get('scheduler_select', "lambdalr")
    if scheduler_select == "lambdalr":
        scheduler_conf = dict(scheduler_conf['conf'])
        lr_lambda = scheduler_conf.pop('lr_lambda')
        lr_lambda = eval(lr_lambda)
        return torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lr_lambda,
            **scheduler_conf,
        )
    else:
        raise NotImplementedError(f"the scheduler policy {scheduler_select}")