import logging

from AMINO.utils.dynamic_import import dynamic_import

def init_optim(model, optim_conf):
    logging.warning(f"using {optim_conf['select']} builder")
    optim_class = dynamic_import(optim_conf['select'])
    parameters = model.parameters()
    return optim_class(parameters, **optim_conf['conf'])