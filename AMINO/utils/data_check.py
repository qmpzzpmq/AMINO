import os
from datetime import datetime
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

def multiple_data_check(datas, datas_len, dim=-2):
    results = list()
    for data, data_len in zip(datas, datas_len):
        results.append(single_data_check(data, data_len, dim=dim))
    return bool(torch.logical_and(torch.tensor(results)).bool())

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
        logging.warning(f"data_len: {data_len}")
        extract_data = torch.index_select(
            data, dim, 
            torch.arange(0, data_len).to(device=data.device),
        )
        result = {
            "nan": tensor_nan_check(extract_data),
            "inf": tensor_inf_check(extract_data),
        }
        temp = [ k for k, v in result.items() if v]
        logging.warning(f"data check result: {result}")
        result['total'] = True if len(temp) > 1 else False
        results.append(result)
    return result

def save_error_tesnsor(data, dir):
    now = datetime.now()
    file_path = f"error{now.strftime('%m.%d.%Y_%H:%M:%S')}.pt"
    logging.warning(f"saving error tensor into {file_path}")
    if not os.path.isfile(file_path):
        torch.save(
            data,
            os.path.join(dir, file_path),
        )