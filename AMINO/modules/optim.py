import logging

from contiguous_params import ContiguousParams

from AMINO.utils.dynamic_import import dynamic_import

def init_optim(model, optim_conf):
    logging.warning(f"using {optim_conf['select']} builder")
    optim_class = dynamic_import(optim_conf['select'])
    if optim_conf.get('contiguous_params', False):
        raise NotImplementedError(
            f"contiguous_params in optim is not test yet"
        )
    parameters = ContiguousParams(model.parameters()) \
        if optim_conf.get('contiguous_params', False) \
        else model.parameters()
    return optim_class(parameters, **optim_conf['conf'])