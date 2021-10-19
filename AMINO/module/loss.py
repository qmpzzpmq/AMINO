import torch

def init_loss(loss_conf):
    loss_class = eval(f"{loss_conf['select']}")
    loss = loss_class(**loss_conf['conf'])
    return loss