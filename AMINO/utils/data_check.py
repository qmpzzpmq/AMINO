import logging

import torch

def tensor_nan_check(data):
    result = torch.isnan(data).sum()
    if result > 0:
        logging.warning(f"tensor contain {result} nan")
    return bool(result.bool())

def tensor_inf_check(data):
    result = torch.isinf(data).sum()
    if result > 0:
        logging.warning(f"tensor contain {result} inf")
    return bool(result.bool())

def single_data_check(data, data_len, dim=-2):
    data = torch.index_select(data, dim, torch.range(0, data_len))
    return tensor_nan_check(data) and tensor_inf_check(data)

def multiple_data_check(data, data_len, dim=-2):
    results = list()
    for data, data_len in zip(datas, datas_len):
        results.append(single_data_check(data, data_len, dim=-2))
    return bool(torch.logical_and(torch.tensor(result)).bool())

def total_check(batch, dim=-2):
    logging.warning(f"DATA SHARP DEBUG:")
    logging.warning(f"data: {batch['feature']['data'].shape}")
    logging.warning(f"label: {batch['label']['data'].shape}")
    logging.warning(f"DATA LEN SHARP DEBUG:")
    logging.warning(f"data: {batch['feature']['len'].shape}")
    logging.warning(f"label: {batch['label']['len'].shape}")
    datas = batch['feature']['data']
    datas_len = batch['feature']['len']
    results = list()
    for data, data_len in zip(datas, datas_len):
        extract_data = torch.index_select(data, dim, torch.range(0, data_len))
        result = {
            "nan": tensor_nan_check(extract_data),
            "inf": tensor_inf_check(extract_data),
        }
        temp = [ k for k, v in results.items() if v]
        if len(temp) > 1:
            logging.warning(f"data {temp}")
            result["total"] = True
        else:
            result["total"] = False
        results.append(result)
    return result