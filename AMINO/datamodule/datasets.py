import logging
import random
import glob

import numpy as np

import torchaudio
import torchaudio.sox_effects as sox_effects
import torch
import torch.utils.data as tdata

from AMINO.utils.dynamic_import import dynamic_import

# copy from https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/dataset_deprecated.py
# add speed perturb when loading wav
# return augmented, sr
def _load_wav_with_speed(wav_file, speed):
    """ Load the wave from file and apply speed perpturbation
    Args:
        wav_file: input feature, T * F 2D
    Returns:
        augmented feature
    """
    if speed == 1.0:
        wav, sr = torchaudio.load(wav_file)
    else:
        sample_rate = torchaudio.backend.sox_io_backend.info(
            wav_file).sample_rate
        # get torchaudio version
        ta_no = torchaudio.__version__.split(".")
        ta_version = 100 * int(ta_no[0]) + 10 * int(ta_no[1])

        if ta_version < 80:
            # Note: deprecated in torchaudio>=0.8.0
            E = sox_effects.SoxEffectsChain()
            E.append_effect_to_chain('speed', speed)
            E.append_effect_to_chain("rate", sample_rate)
            E.set_input_file(wav_file)
            wav, sr = E.sox_build_flow_effects()
        else:
            # Note: enable in torchaudio>=0.8.0
            wav, sr = sox_effects.apply_effects_file(
                wav_file,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
    return wav, sr

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
            speed_perturb=[1.0, 1.1, 0.9],
            speed_perturb_weight=[1, 1, 1]
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
        self.speed_perturb = speed_perturb
        self.speed_perturb_weight = speed_perturb_weight
        self.preprocess_func = None

    def set_preprocesses(self, preprocesses_func):
        self.preprocess_func = preprocesses_func

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # read
        try:
            # data, fs = torchaudio.load(self.file_list[index], channels_first=True)
            speed = random.choices(
                self.speed_perturb,
                self.speed_perturb_weight,
                k=1
            )[0] if self.speed_perturb is not None else 1.0
            data, fs = _load_wav_with_speed(
                self.file_list[index],
                speed,
            )
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