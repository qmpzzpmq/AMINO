import os
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import yaml

import pytorch_lightning as pl

from AMINO.configs.configs import TRAIN_CONFIG, register_OmegaConf_resolvers
from AMINO.datamodule.datamodule import init_datamodule
from AMINO.utils.callbacks import init_callbacks
from AMINO.utils.loggers import init_loggers
from AMINO.modules.modules import init_module

@hydra.main(
    config_path=os.path.join(os.getcwd(), 'conf'),
    config_name="example.yaml",
)
def main(read_cfg) -> None:
    dft_cfg = OmegaConf.structured(TRAIN_CONFIG)
    # only for dev
    OmegaConf.save(
        config=dft_cfg,
        f=os.path.join(
            hydra.utils.get_original_cwd(), 'conf', 'default.yaml',
        )
    )
    
    cfg = OmegaConf.merge(dft_cfg, read_cfg)
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    logging.basicConfig(
        level=cfg.logging.level,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
    )
    hydra_config = HydraConfig.get()

    callbacks = init_callbacks(cfg['callbacks'])
    loggers = init_loggers(cfg['loggers'])
    datamodule = init_datamodule(cfg['datamodule'])
    module = init_module(cfg['module'])
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **cfg['trainer'],
    )

    logging.warning(
        f"start {hydra_config.job.name} job at dir {os.getcwd()}",
    )
    trainer.fit(module, datamodule=datamodule)
    logging.warning("done")

if __name__ == "__main__":
    register_OmegaConf_resolvers()
    main()
