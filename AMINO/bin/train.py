import os
import logging

import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl
import torch
import torch.utils.data as tdata

from AMINO.configs.configs import TRAIN_CONFIG
from AMINO.data.datamodule import init_datamodule
from AMINO.utils.callbacks import init_callbacks
from AMINO.utils.loggers import init_loggers

#debug
from AMINO.configs.datamodule import PREPORCESSES
from AMINO.configs.common import AMINO_CONF

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
    # raaec = RAAEC(cfg['module'], cfg['optim'], cfg['loss'])
    
    # trainer = pl.Trainer(
    #     callbacks=callbacks,
    #     logger=loggers,
    #     **cfg['trainer'],
    # )
    # trainer.fit(raaec, datamodule=dm)


if __name__ == "__main__":
    main()