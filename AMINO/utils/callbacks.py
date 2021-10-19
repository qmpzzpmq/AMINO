import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks

def init_callbacks(callbacks_conf):
    callbacks = []
    if callbacks_conf.get("progressbar", False):
        callbacks.append(
            pl_callbacks.progress.ProgressBar(
                **callbacks_conf['progressbar_conf']
            )
        )

    if callbacks_conf.get("modelcheckpoint", False):
        callbacks.append(
            pl_callbacks.model_checkpoint.ModelCheckpoint(
                **callbacks_conf['modelcheckpoint_conf']
            )
        )
    
    if callbacks_conf.get("earlystopping", False):
        callbacks.append(
            pl_callbacks.early_stopping.EarlyStopping(
                **callbacks_conf['earlystopping_conf']
            )
        )

    if callbacks_conf.get('gpu_stats', False):
        callbacks.append(
            pl_callbacks.gpu_stats_monitor.GPUStatsMonitor(
                **callbacks_conf['gpu_stats_conf']
            )
        )
    if callbacks_conf.get('lr_monitor', False):
        callbacks.append(
            pl_callbacks.lr_monitor.LearningRateMonitor(
                **callbacks_conf['lr_monitor_conf']
            )
        ) 

    return callbacks

@hydra.main(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    callbacks = init_callbacks(cfg['callbacks'])
    print(f"callbacks: {callbacks}")

if __name__ == "__main__":
    unit_test()