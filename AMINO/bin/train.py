import os
import logging

import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

from AMINO.configs.configs import TRAIN_CONFIG, register_OmegaConf_resolvers
from AMINO.datamodule.datamodule import AMINODataModule
from AMINO.modules.classifier import AMINO_CLASSIFIER
from AMINO.utils.dynamic_import import dynamic_import
from AMINO.utils.init_object import init_object, init_list_object
from AMINO.utils.configs import cfg_process

@hydra.main(
    config_path=os.path.join(os.getcwd(), 'conf'),
    config_name="melspectro_feature_clean.yaml",
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
    # temp_cfg = OmegaConf.to_yaml(read_cfg)
    # cfg = OmegaConf.structured(TRAIN_CONFIG(**read_cfg))

    hydra_config = hydra.core.hydra_config.HydraConfig.get()

    callbacks = init_list_object(cfg['callbacks'])
    loggers = init_list_object(cfg['loggers'])
    datamodule = AMINODataModule(cfg['datamodule'])
    if issubclass(dynamic_import(cfg['module']['select']), AMINO_CLASSIFIER):
        datamodule.setup() # dev
        num_classes = datamodule.get_num_classes()
        if "num_classes" in cfg['pipeline_size'] and \
                type(cfg['pipeline_size']['num_classes']) == int:
            assert num_classes == cfg['module_flexible_size']['num_classes'], \
                f"the num_classes {num_classes} in dataset not equals \
                    num_classes {cfg['pipeline_size']['num_classes']} in configs"
        else:
            cfg['pipeline_size']['num_classes'] = num_classes
    OmegaConf.save(config=cfg, f=f'{hydra_config.job.name}.yaml')
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')

    module = init_object(cfg['module'])
    pl.utilities.model_summary.summarize(module, max_depth=-1)

    cfg = cfg_process(cfg)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **cfg["trainer"],
    )

    logging.warning(
        f"start {hydra_config.job.name} job at dir {os.getcwd()}",
    )
    if cfg.trainer.auto_scale_batch_size is not None:
        datamodule.batch_size = datamodule.datamodule_conf[
            'dataloaders']['train']['batch_size']
        result = trainer.tune(
            module,
            datamodule=datamodule,
        )
        del datamodule.batch_size
        datamodule.datamodule_conf.dataloaders.train['batch_size'] = result[
            "scale_batch_size"]
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.checkpoint)
    logging.warning("done")

if __name__ == "__main__":
    register_OmegaConf_resolvers()
    main()
