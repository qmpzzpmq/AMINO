from AMINO.utils.dynamic_import import dynamic_import
from AMINO.modules.nets.decoder import AMINO_CLASSIFIER
from AMINO.modules.autoencoder import AMINO_AUTOENCODER

def cfg_process(cfg):
    for batch_name in [
        "limit_train_batches", "limit_val_batches", 
        "limit_test_batches", "limit_predict_batches"
    ]:
        if cfg["trainer"][batch_name] > 1.0:
            cfg["trainer"][batch_name] = int(cfg["trainer"][batch_name])
        else:
            cfg["trainer"][batch_name] = float(cfg["trainer"][batch_name])
    return cfg

def is_classifier(net_cfg):
    if issubclass(dynamic_import(net_cfg["select"]), AMINO_AUTOENCODER):
        return False
    decoders_conf = net_cfg["conf"]["decoders"]
    for v in decoders_conf.values():
        if issubclass(dynamic_import(v["select"]), AMINO_CLASSIFIER):
            return True
    return False
