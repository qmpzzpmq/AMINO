import logging

from omegaconf import OmegaConf

from AMINO.bin.train import common_prepare

def main(cfg) -> None:
    trainer, module, datamodule = common_prepare(cfg)
    if not OmegaConf.select(cfg, "expbase.checkpoint"):
        logging.warning(
            "the expbase.checkpoint is empty, which means the init model will be used for validate"
        )
    trainer.validate(
        module, datamodule=datamodule,
        ckpt_path=OmegaConf.select(cfg, "expbase.checkpoint")
    )
    logging.warning("done")