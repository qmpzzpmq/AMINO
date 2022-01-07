import logging
import os
import random
import glob
import csv
from typing import OrderedDict

import torchaudio
import torchaudio.sox_effects as sox_effects
import torch

from AMINO.utils.dynamic_import import dynamic_import
from AMINO.utils.init_object import init_object

# copy from https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/dataset_deprecated.py
# add speed perturb when loading wav
# return augmented, sr
def _load_wav_with_speed(wav_file, speed, start=0, end=-1):
    """ Load the wave from file and apply speed perpturbation
    Args:
        wav_file: input feature, T * F 2D
    Returns:
        augmented feature
    """
    if speed == 1.0:
        wav, sr = torchaudio.load(
            wav_file, 
            frame_offset=start, num_frames=end-start
        )
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

def toyadmos2_file_list_generator(
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

def dcase2020_task2_file_list_generator(
        path,
        prefix_normal="normal",
        prefix_anormal="anomaly",
        ext="wav",
    ):
    files_list = []
    for prefix in [prefix_normal, prefix_anormal]:
        files_list.append(
            glob.glob(
                "{path}/{prefix}_*.{ext}".format(
                    path=path,
                    prefix=prefix,
                    ext=ext,
                )
            )
        )
    return files_list

def audioset_csv_reader(csv_path, dict_key):
    data = []
    with open(csv_path, 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            if (not row[0].startswith("#")) and \
                    (len(row) == len(dict_key)) and \
                    (row[0] != "index"):
                item_dict = {}
                for key, value in zip(dict_key, row):
                    item_dict[key] = value.strip(" ").strip("\"")
                data.append(item_dict)
    return data

def audioset_label_dict_builder(data_list):
    label_dict = {}
    for item in data_list:
        id = item['mid']
        if id in label_dict:
            raise ValueError(f"{id} already exists in dict")
        token = int(item['index'])
        label_dict[id] = token
    values = torch.tensor(list(label_dict.values()), dtype=torch.int)
    assert (values.min() == 0) and (values.max() + 1 == len(values)), \
        f"there is some empty {token}"
    return label_dict

def audioset_tokener(data_list, label_dict, data_dir):
    token_data_list = []
    for item in data_list:
        token_data_list.append({
            "path": os.path.join(data_dir, f"{item['YTID']}.wav"),
            "start": item['start_seconds'],
            "end": item['end_seconds'],
            "token": torch.tensor(
                [label_dict[x] for x in item['positive_labels'].split(',')],
            )
        })
    return token_data_list

class AMINO_DATASET(torch.utils.data.Dataset):
    def __init__(
            self,
            fs=16000,
            mono_channel="mean",
            speed_perturb=[1.0, 1.1, 0.9],
            speed_perturb_weight=[1, 1, 1],
        ):
        self.fs = fs
        self.speed_perturb = speed_perturb
        self.speed_perturb_weight = speed_perturb_weight
        self.preprocess_func = None
        if mono_channel == "mean":
            self.mono_func = lambda x: torch.mean(x, dim=0).unsqueeze(0)
        elif mono_channel.isdigit() and len(mono_channel) == 1:
            self.mono_func = lambda x: x[int(mono_channel), :].unsqueeze(0)
        else:
            raise ValueError(
                f"the mono_channel setting wrong, please read the help in AMINO/configs/datamodule.py"
            )

    def set_preprocesses(self, preprocesses_func):
        self.preprocess_func = preprocesses_func
    def standard_read(self, path, start=0, end=-1):
        try:
            # data, fs = torchaudio.load(self.file_list[index], channels_first=True)
            speed = random.choices(
                self.speed_perturb,
                self.speed_perturb_weight,
                k=1
            )[0] if self.speed_perturb is not None else 1.0
            data, fs = _load_wav_with_speed(
                path,
                speed,
                start,
                end,
            )
        except Exception as e:
            logging.warning(
                f"Something wrong with resample {path}, skip the utt"
            )
            logging.warning(e)
            return None
        if self.preprocess_func is not None:
            try:
                data = self.preprocess_func([data, fs])
            except Exception as e:
                logging.warning(
                    f"Something wrong with preprocess {path}, skip the preprocess"
                )
                logging.warning(e)
                return data
        return data

class AMINO_ConcatDataset(torch.utils.data.ConcatDataset):
    def set_preprocesses(self, preprocesses_func):
        for i in range(len(self.datasets)):
            self.datasets[i].preprocess_func = preprocesses_func
    def get_num_classes(self):
        try:
            num_classes = torch.tensor(
                [dataset.get_num_classes() for dataset in self.datasets]
            )
            assert num_classes.min() == num_classes.max(), \
                f"the number of classes in each dataset is not same"
        except Exception as e:
            logging.warning(
                f"Something wrong with get_num_classes, return None"
            )
            return None
        return num_classes.max()
        

class ADMOS_DATASET(AMINO_DATASET):
    def __init__(
            self,
            path,
            format="ToyADMOS2",
            prefix_normal="normal",
            prefix_anomaly="anomaly",
            ext="wav",
            mono_channel="mean",
            fs=16000,
            speed_perturb=[1.0, 1.1, 0.9],
            speed_perturb_weight=[1, 1, 1],
    ):
        super().__init__(fs, mono_channel, speed_perturb, speed_perturb_weight)
        if format == "ToyADMOS2":
            normal_files, anormal_files = toyadmos2_file_list_generator(
                path, prefix_normal, prefix_anomaly, ext
            )
        elif format == "dcase2020_task2":
            normal_files, anormal_files = dcase2020_task2_file_list_generator(
                path, prefix_normal, prefix_anomaly, ext
            )
        else:
            raise NotImplementedError(f"{format} not support")
        self.file_list = normal_files + anormal_files
        self.label = torch.cat([
            torch.full([len(normal_files)], True, dtype=torch.bool),
            torch.full([len(anormal_files)], False, dtype=torch.bool),
        ], dim=0)


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        data = self.standard_read(self.file_list[index])
        if data is None:
            return None
        label = torch.tensor([self.label[index]])
        return data, label 


class AUDIOSET_DATASET(AMINO_DATASET):
    def __init__(
        self,
        data_dir,
        dataset="balanced_train",
        mono_channel="mean",
        fs=16000,
        speed_perturb=[1.0, 1.1, 0.9],
        speed_perturb_weight=[1, 1, 1],
    ):
        super().__init__(fs, mono_channel, speed_perturb, speed_perturb_weight)
        label_list = audioset_csv_reader(
            os.path.join(data_dir, "metadata", "class_labels_indices.csv"),
            ["index", "mid", "display_name"],
        )
        audio_dir = os.path.join(data_dir, "audios")
        if dataset == "balanced_train":
            file_name = os.path.join(
                data_dir, "metadata", "balanced_train_segments.csv",
            )
            segments_dir = os.path.join(audio_dir, "balanced_train_segments")
        elif dataset == "unbalance_train":
            file_name = os.path.join(
                data_dir, "metadata", "unbalanced_train_segments.csv"
            )
            segments_dir = os.path.join(audio_dir, "unbalanced_train_segments")
        elif dataset == "eval":
            file_name = os.path.join(
                data_dir, "metadata", "eval_segments.csv"
            )
            segments_dir = os.path.join(audio_dir, "eval_segments")
        else:
            raise ValueError(f"""
                dataset shoule be set between balanced_train/unbalance_train/eval,
                now it is {dataset}.
            """)
        assert os.path.isdir(segments_dir), \
            "the segment dir {segments_dir} donesn't exist"
        data_list = audioset_csv_reader(
            file_name,
            ["YTID", "start_seconds", "end_seconds", "positive_labels"],
        )
        label_dict = audioset_label_dict_builder(label_list)
        self.num_classes = len(label_dict.values())
        self.data_list = audioset_tokener(data_list, label_dict, segments_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        data = self.standard_read(item['path'], item['start'], item['end'])
        if data is None:
            return None
        label = item['token']
        return data, label
    
    def get_num_classes(self):
        return self.num_classes
