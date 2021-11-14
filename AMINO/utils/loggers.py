import os

import pytorch_lightning as pl

from AMINO.utils.dynamic_import import dynamic_import

def init_loggers(loggers_conf):
    loggers = []
    for logger_conf in loggers_conf:
        logger_class = dynamic_import(logger_conf['select'])
        loggers.append(
            logger_class(**logger_conf['conf'])
        )
    return loggers
