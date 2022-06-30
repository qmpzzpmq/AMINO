import logging
from copy import deepcopy

from AMINO.utils.dynamic_import import dynamic_import

def _init_optim(model, optim_conf):
    logging.warning(f"using {optim_conf}")
    optim_class = dynamic_import(optim_conf['select'])
    if optim_conf.get("backend", "torch") == "torch":
        return optim_class(
            model.parameters(), **optim_conf['conf']
        )
    else:
        # https://github.com/pyro-ppl/pyro/blob/dev/pyro/optim/lr_scheduler.py
        if "optimizer" in optim_conf['conf'] \
                and "optim_args" in optim_conf['conf']:
            temp_optim_conf = dict(optim_conf['conf'])
            temp_optim_class = dynamic_import(
                temp_optim_conf.pop("optimizer")
            )
            optim_args = temp_optim_conf.pop("optim_args")
            temp_dict = {
                "optimizer": temp_optim_class,
                "optim_args": dict(optim_args),
                **temp_optim_conf,
            }
            return optim_class(temp_dict)
        else:
            return optim_class(dict(optim_conf['conf']))

def init_optim(model, optim_conf):
    logging.warning(f"using {optim_conf}")
    optim_class = dynamic_import(optim_conf['select'])
    return optim_class(
        model.parameters(), **optim_conf['conf']
    )
