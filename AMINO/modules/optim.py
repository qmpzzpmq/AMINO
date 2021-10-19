import logging

import torch
from contiguous_params import ContiguousParams

def init_optim(model, optim_conf):
    optim_select = optim_conf.get('select', "optim.Adam")
    logging.warning(f"using {optim_select} builder")
    optim_class = eval(optim_select)
    parameters = ContiguousParams(model.parameters()) \
        if optim_conf.get('contiguous_params', False) \
        else model.parameters()
    return optim_class(parameters, **optim_conf['conf'])