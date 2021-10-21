import logging
import os
import glob

import torchaudio
import torch
import torch.nn as nn
import torch.utils.data as tdata

def file_list_generator(path, ext="wav"):
    # generate training list
    training_list_path = os.path.abspath("{dir}/*.{ext}".format(dir=path, ext=ext))
    files = sorted(glob.glob(training_list_path))
    return files

class TOYADMOS2_DATASET(tdata.Dataset):
    def __init__(
            self,
            path,
            ext="wav",
    ):
        super().__init__()
        self.file_list = file_list_generator(path, ext)
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
        if self.preprocess_func is not None:
            try:
                data = self.preprocess_func([data, fs])
            except Exception as e:
                logging.warning(
                    f"Something wrong with preprocess {self.file_list[index]}, skip the preprocessit"
                )
                logging.warning(e)
                return data
        return data
        

def init_dataset(dataset_conf):
    dataset_class = eval(dataset_conf['select'])
    dataset = dataset_class(**dataset_conf['conf'])
    return dataset