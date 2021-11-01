import os
import logging

import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import pytorch_lightning as pl

from AMINO.configs.configs import TRAIN_CONFIG, register_OmegaConf_resolvers
from AMINO.datamodule.datamodule import init_datamodule
from AMINO.utils.callbacks import init_callbacks
from AMINO.utils.loggers import init_loggers
from AMINO.modules.modules import init_module

@hydra.main(
    config_path=os.path.join(os.getcwd(), 'conf'),
    config_name="spectral.yaml",
)
def main(read_cfg) -> None:
    dft_cfg = OmegaConf.structured(TRAIN_CONFIG)

    cfg = OmegaConf.merge(dft_cfg, read_cfg)
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    logging.basicConfig(
        level=cfg.logging.level,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
    )

    datamodule = init_datamodule(cfg['datamodule'])
    module = init_module(cfg['module'])

    module.feature_statistics_init(cfg.module.conf.net_conf.conf.feature_dim)
    module.predict_step = module.feature_statistics
    datamodule.setup()
    datamodule.transform2device['after']['predict'] = False
    datamodule.predict_dataloader = datamodule.val_dataloader
    datamodule.transform['after']['predict'] = datamodule.transform['after']['val']
    trainer = pl.Trainer(**cfg['trainer'],)
    trainer.predict(module, datamodule)
    module.feature_statistics_end()


if __name__ == "__main__":
    register_OmegaConf_resolvers()
    main()
