import os
import logging

import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

from AMINO.configs.configs import TRAIN_CONFIG, register_OmegaConf_resolvers
from AMINO.datamodule.datamodule import init_datamodule
from AMINO.modules.modules import init_module
from AMINO.utils.dynamic_import import path_convert
from AMINO.datamodule.preprocess import TrainDataAugment
from AMINO.utils.dynamic_import import dynamic_import

@hydra.main(
    config_path=os.path.join(os.getcwd(), 'conf'),
    config_name="melspectro_feature_clean.yaml",
)
def main(read_cfg) -> None:
    dft_cfg = OmegaConf.structured(TRAIN_CONFIG)

    cfg = OmegaConf.merge(dft_cfg, read_cfg)
    using_set = cfg.feature_statistics.get("set", "val")

    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    logging.basicConfig(
        level=cfg.logging.level,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
    )

    feature_dim = cfg.module.conf.net_conf.conf.feature_dim
    cmvn_path = cfg.module.conf.net_conf.conf.cmvn_path
    cfg.module.select = "AMINO.modules.base_module:AMINO_MODULE"
    cfg.module.conf = {}
    cfg.trainer['strategy'] = 'ddp'
    cfg.trainer['accelerator'] = 'cpu'
    cfg.datamodule.datasets.train.conf.speed_perturb=None
    temp_train_after_transform = []
    for i in cfg.datamodule.after_transform.train:
        if not issubclass(dynamic_import(i['select']), TrainDataAugment):
            temp_train_after_transform.append(i)
    cfg.datamodule.after_transform.train = temp_train_after_transform

    datamodule = init_datamodule(cfg['datamodule'])
    module = init_module(cfg['module'])

    module.feature_statistics_init(feature_dim)
    module.predict_step = module.feature_statistics
    datamodule.setup()
    datamodule.transform2device['after']['predict'] = False
    datamodule.predict_dataloader = getattr(datamodule, f"{using_set}_dataloader")
    datamodule.transform['after']['predict'] = datamodule.transform['after'][using_set]
    trainer = pl.Trainer(**cfg['trainer'],)
    trainer.predict(module, datamodule)
    module.feature_statistics_end(
        dump_path=path_convert(cmvn_path),
    )


if __name__ == "__main__":
    register_OmegaConf_resolvers()
    main()
