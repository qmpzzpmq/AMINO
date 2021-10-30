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
                    ext=ext,
                )
            )
        )
    return files_list

class AMINO_ConcatDataset(torch.utils.data.ConcatDataset):
    def set_preprocesses(self, preprocesses_func):
        for i in range(len(self.datasets)):
            self.datasets[i].preprocess_func = preprocesses_func

class TOYADMOS2_DATASET(tdata.Dataset):
    def __init__(
            self,
            path,
            prefix_normal="normal",
            prefix_anomaly="anomaly",
            ext="wav",
            mono_channel="mean",
            fs=16000,
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
        self.fs = fs
        if mono_channel == "mean":
            self.mono_func = lambda x: torch.mean(x, dim=0).unsqueeze(0)
        elif mono_channel.isdigit() and len(mono_channel) == 1:
            self.mono_func = lambda x: x[int(mono_channel), :].unsqueeze(0)
        else:
            raise ValueError(
                f"the mono_channel setting wrong, please read the help in AMINO/configs/datamodule.py"
            )
        self.preprocess_func = None

    def set_preprocesses(self, preprocesses_func):
        self.preprocess_func = preprocesses_func

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # read
        try:
            data, fs = torchaudio.load(self.file_list[index], channels_first=True)
            label = torch.tensor([self.label[index]])
        except Exception as e:
            logging.warning(
                f"Something wrong with read {self.file_list[index]}, skip the utt"
            )
            logging.warning(e)
            return None
        # channel build && resample
        try:
            data = self.mono_func(data)
            if fs != self.fs:
                data = torchaudio.functional.resample(data, fs, self.fs)
        except Exception as e:
            logging.warning(
                f"Something wrong with resample {self.file_list[index]}, skip the utt"
            )
            logging.warning(e)
            return None
        # preprocess
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

def init_datasets(datasets_conf):
    datasets = []
    for dataset_conf in datasets_conf:
        dataset = init_dataset(dataset_conf)
        if dataset is None:
            continue
        else:
            datasets.append(dataset)
    if len(datasets) > 0:
        return AMINO_ConcatDataset(datasets)
    else:
        return None