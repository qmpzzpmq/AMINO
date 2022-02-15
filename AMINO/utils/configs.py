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