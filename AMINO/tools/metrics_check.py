import sys
import logging
import argparse

import numpy as np
import torch

from sklearn import metrics as smetrics
from torchmetrics import functional as tmetics

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["ALL", "INFO", "DEBUG", "WARN"],
        default="INFO",
    )
    return parser

def common_test(
        true, pred,
        s_test_func, t_test_func,
        s_test_func_kargs={}, t_test_func_kargs={},
    ):
    sscore = s_test_func(
        true.cpu().numpy(), pred.cpu().numpy(),
        **s_test_func_kargs,
    )
    tscore = t_test_func(pred, true, **t_test_func_kargs)
    logging.debug(f"sscore: {sscore}")
    logging.debug(f"tscore: {tscore}")
    if type(tscore) == list:
       tscore = torch.tensor(tscore)
    return np.isclose(sscore, tscore.cpu().numpy())

def acc_test(true, pred):
    logging.debug(f"acc")
    return common_test(true, pred, smetrics.accuracy_score, tmetics.accuracy)

def auc_test(true, pred):
    logging.debug(f"auc")
    return common_test(
        true, pred, 
        smetrics.roc_auc_score, tmetics.auroc,
        t_test_func_kargs={"num_classes": int(pred.size(-1))},
    )

def map_test_noaverage(true, pred):
    logging.debug(f"mag")
    return common_test(
        true, pred,
        smetrics.average_precision_score, tmetics.average_precision,
        s_test_func_kargs={"average": None},
        t_test_func_kargs={"average": None, "num_classes": int(pred.size(-1))},

    )

def map_test(true, pred):
    logging.debug(f"mag")
    return common_test(
        true, pred,
        smetrics.average_precision_score, tmetics.average_precision,
        s_test_func_kargs={},
        t_test_func_kargs={"num_classes": int(pred.size(-1))},

    )

def full_tset(true, pred):
    # acc = acc_test(true, pred)
    acc = 0
    auc = auc_test(true, pred)
    map = map_test(true, pred)
    return {"acc": acc, "auc": auc, "map": map}

def main(cmd_args):
    parser = get_parser()
    args = parser.parse_args(cmd_args)
    logging_level = eval("logging." + args.logging_level)
    logging.basicConfig(
        level=logging_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # pred = torch.tensor([
    #     [0.1, 0.3, 0.5],
    #     [0.5, 0.3, 0.1],
    #     [0.1, 0.3, 0.5],
    # ])
    # true = torch.tensor([
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 0, 1],
    # ])
    pred = torch.tensor([
        [0.75, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.75, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.75, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.75, 0.05],
    ])
    true = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
    ])
    # result = full_tset(true, pred)
    result = map_test_noaverage(true, pred)
    print(result)
    result = map_test(true, pred)
    print(result)

if __name__ == "__main__":
    main(sys.argv[1:])