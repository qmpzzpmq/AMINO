import logging
import os
import glob

import numpy as np

import torchaudio
import torch
import torch.utils.data as tdata

from AMINO.utils.dynamic_import import dynamic_import

def file_list_generator(
        path,
        prefix_normal="normal",
        prefix_anormal="anomaly",
        ext="wav",
    ):
    files_list = []
    for prefix in [prefix_normal, prefix_anormal]:
        files_list.append(
            glob.glob(
                "{path}/*_{prefix}_*.{ext}".format(
                    path=path,
                    prefix=prefix,
                    ext=ext
                )
            )
        )
    return files_list

class TOYADMOS2_DATASET(tdata.Dataset):
    def __init__(
            self,
            path,
            prefix_normal="normal",
            prefix_anomaly="anomaly",
            ext="wav",
    ):
        super().__init__()
        normal_files, anormal_files = file_list_generator(
            path, prefix_normal, prefix_anomaly, ext)
        self.file_list = normal_files + anormal_files
        self.label = np.concatenate(
            [
                np.full([len(normal_files),], True, dtype=bool),
                np.full([len(anormal_files),], False, dtype=bool),
            ],
            axis=0,
        )
        self.preprocess_func = None

    def set_preprocesses(self, preprocesses_func):
        self.preprocess_func = preprocesses_func

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        try:
            data, fs = torchaudio.load(self.file_list[index], channels_first=True)
        except Exception as e:
            logging.warning(
                f"Something wrong with read {self.file_list[index]}, skip the utt"
            )
            logging.warning(e)
            return None
        label = torch.tensor([self.label[index]])
        if self.preprocess_func is not None:
            try:
                data = self.preprocess_func([data, fs])
            except Exception as e:
                logging.warning(
                    f"Something wrong with preprocess {self.file_list[index]}, skip the preprocess"
                )
                logging.warning(e)
                return data, label
        return data, label
        

def init_dataset(dataset_conf):
    if dataset_conf is None:
        return None
    dataset_class = dynamic_import(dataset_conf['select'])
    dataset = dataset_class(**dataset_conf['conf'])
    return dataset