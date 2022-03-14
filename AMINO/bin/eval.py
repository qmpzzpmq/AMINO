import os
import logging
import copy

import hydra
from omegaconf import OmegaConf

from AMINO.utils.resolvers import register_OmegaConf_resolvers
from AMINO.bin.train import common_prepare

@hydra.main(
    config_path=os.path.join(os.getcwd(), 'conf'),
)
def main(org_cfg) -> None:
    cfg = dict(copy.deepcopy(org_cfg))
    del cfg["loggers"], cfg["callbacks"]
    cfg = OmegaConf.create(cfg)

    trainer, module, datamodule = common_prepare(cfg)
    if not OmegaConf.select(cfg, "expbase.checkpoint"):
        logging.warning(
            "the expbase.checkpoint is empty, which means the init model will be used for validate"
        )
    ckpt_path = hydra.utils.to_absolute_path(
        OmegaConf.select(cfg, "expbase.checkpoint")
    )
    trainer.validate(
        module, datamodule=datamodule, ckpt_path=ckpt_path,
    )
    # callback_metrics = trainer.callback_metrics
    # progress_bar_metrics = trainer.progress_bar_metrics
    # logged_metrics = trainer.logged_metrics
    logging.warning("done")

if __name__ == "__main__":
    register_OmegaConf_resolvers()
    main()