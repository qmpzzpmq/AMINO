import os
import logging

import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

from AMINO.configs.configs import TRAIN_CONFIG
from AMINO.data.datamodule import init_datamodule
from AMINO.utils.callbacks import init_callbacks
from AMINO.utils.loggers import init_loggers
from AMINO.modules.modules import init_module


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'conf'),
    config_name="default.yaml",
)
def main(read_cfg) -> None:
    dft_cfg = OmegaConf.structured(TRAIN_CONFIG)
    cfg = OmegaConf.merge(dft_cfg, read_cfg)
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    logging.basicConfig(
        level=cfg.logging.level,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    callbacks = init_callbacks(cfg['callbacks'])
    loggers = init_loggers(cfg['loggers'])
    dm = init_datamodule(cfg['datamodule'])
    module = init_module(cfg['module'])
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **cfg['trainer'],
    )
    trainer.fit(module, datamodule=dm)
    logging.warning("done")


if __name__ == "__main__":
    main()