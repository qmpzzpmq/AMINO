import os
import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers

def init_loggers(loggers_conf):
    loggers = []
    if loggers_conf.get('tensorboard', False):
        loggers.append(
            pl.loggers.TensorBoardLogger(**loggers_conf['tensorboard_conf'])
        )

    if loggers_conf.get('wandb', False):
        wandb_dir = loggers_conf['wandb_conf']['save_dir']
        if not os.path.isdir(wandb_dir):
            os.makedirs(wandb_dir)
        loggers.append(
            pl.loggers.wandb.WandbLogger(**loggers_conf['wandb_conf'])
        )

    if loggers_conf.get('neptune',False):
        loggers.append(
            pl.loggers.neptune.NeptuneLogger(**loggers_conf['neptune_conf'])
        )

    return loggers

@hydra.main(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    loggers = init_loggers(cfg['loggers'])
    print(f"loggers: {loggers}")

if __name__ == "__main__":
    unit_test()