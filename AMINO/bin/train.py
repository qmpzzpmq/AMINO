import os
import logging

import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

from AMINO.utils.resolvers import register_OmegaConf_resolvers
from AMINO.datamodule.datamodule import AMINODataModule
from AMINO.utils.init_object import init_object, init_list_object
from AMINO.utils.configs import cfg_process
from AMINO.utils.datamodule import get_auto_batch_size

def common_prepare(cfg):
    # data prepare
    datamodule = AMINODataModule(cfg['datamodule'])
    num_classes = datamodule.get_num_classes()
    if num_classes :
        if "num_classes" in cfg['pipeline_size'] and \
                type(cfg['pipeline_size']['num_classes']) == int:
            assert num_classes == cfg['module_flexible_size']['num_classes'], \
                f"the {num_classes=} in dataset not equals \
                    num_classes {cfg['pipeline_size']['num_classes']=} in configs"
        else:
            cfg['pipeline_size']['num_classes'] = num_classes
        logging.info(f"num_classes: {num_classes}")

    # module prepare
    module = init_object(cfg['module'])
    pl.utilities.model_summary.summarize(module, max_depth=-1)

    # training utilities prepare
    callbacks = init_list_object(cfg['callbacks']) if "callbacks" in cfg else None
    loggers = init_list_object(cfg['loggers']) if "loggers" in cfg else None

    # trainer prepare
    cfg = cfg_process(cfg)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **cfg["trainer"],
    )

    # trainer prepare
    if cfg.trainer.auto_scale_batch_size:
        datamodule.datamodule_conf.dataloaders.train['batch_size'] = \
            get_auto_batch_size(module, datamodule, trainer)
    assert cfg.trainer.strategy.startswith("ddp")

    if not OmegaConf.select(cfg, "expbase.seed"):
        seed = cfg["expbase"]["seed"]
        logging.info(f"seed all seed to {seed}")
        pl.utilities.seed.seed_everything(seed)
    datamodule.set_replace_sampler_ddp(cfg.trainer.replace_sampler_ddp)
    return trainer, module, datamodule

@hydra.main(
    config_path=os.path.join(os.getcwd(), 'conf'),
)
def main(cfg) -> None:
    trainer, module, datamodule = common_prepare(cfg)
    # save config and start train
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    OmegaConf.save(config=cfg, f=f'{hydra_config.job.name}.yaml')
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    logging.warning(
        f"start {hydra_config.job.name} job at dir {os.getcwd()}",
    )
    trainer.fit(
        module, datamodule=datamodule,
        ckpt_path=OmegaConf.select(cfg, "expbase.checkpoint")
    )
    logging.warning("done")

if __name__ == "__main__":
    register_OmegaConf_resolvers()
    main()
