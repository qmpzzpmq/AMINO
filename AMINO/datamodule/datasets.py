import logging
import os
import random
import glob
import csv
import json
import multiprocessing as mp
import tempfile

from tqdm import tqdm
import hydra

import torchaudio
import torch

def load_wav_with_speed_resample(wav_file, speed, fs=None, start=0, end=-1):
    if start == 0 and end == -1:
        wav, sr = torchaudio.backend.sox_io_backend.load(wav_file)
    else:
        sr = torchaudio.backend.sox_io_backend.info(
            wav_file).sample_rate
        frame_offset = int(sr * start)
        num_frames = int(sr * (end-start))
        wav, sr = torchaudio.backend.sox_io_backend.load(
            wav_file, frame_offset=frame_offset, num_frames=num_frames,
        )
    effects = []
    if not (speed == 1.0):
        effects.append(['speed', str(speed)])
    if (fs is not None) and (fs != sr):
        effects.append(['rate', str(fs)])
    if len(effects) > 0:
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(
            wav, sr, effects
        )
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

def rebuild_row_list(row, out_len):
    row_len = len(row)
    if row_len != out_len:
        return row[0:out_len-1]+ [",".join(row[out_len-1:])]
    return row

def audioset_csv_reader(csv_path, dict_key):
    data = []
    len_dict = len(dict_key)
    with open(csv_path, 'r') as fin:
        # incase some empty line
        reader = csv.reader((line.replace('\0','') for line in fin))
        for row in reader:
            if (not row[0].startswith("#")) and \
                    (row[0] != "index"):
                row = rebuild_row_list(row, len_dict)
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

def list_split(inlist, num_split):
    chunked_list = list()
    chunk_size = len(inlist) // ( num_split -1 )
    for i in range(0, len(inlist), chunk_size):
        chunked_list.append(inlist[i : i+chunk_size])
    return chunked_list

def list_merge(inlist):
    outlist = list()
    for chunk_list in inlist:
        outlist += chunk_list
    return outlist

def file_merge(file_names, out_file):
    temp_file_names = " ".join(file_names)
    os.system(f"cat {temp_file_names} > {out_file}")

def audioset_tokener(
        data_list, label_dict, data_dir, subdir,
        tqdm_config={"position": 0, "disable": False},
    ):
    token_data_list = []
    postion = tqdm_config['position']
    with tqdm(data_list, **tqdm_config) as pbar:
        for i, item in enumerate(pbar):
            if subdir:
                audio_paths = glob.glob(
                    os.path.join(data_dir, "*", f"Y{item['YTID']}.wav")
                )
                if len(audio_paths) == 0:
                    logging.debug(f"{item['YTID']} not exists")
                    continue
                elif len(audio_paths) > 1:
                    raise ValueError(
                        f"same ID {item['YTID']} should only exists onse"
                    )
                audio_path = audio_paths[0]
            else:
                audio_path = os.path.join(data_dir, f"Y{item['YTID']}.wav")
                if not os.path.isfile(audio_path):
                    logging.debug(f"{item['YTID']} not exists")
                    continue
            pbar.set_description(
                f"{postion} effective rate: {(len(token_data_list))/(i+1)}"
            )
            token_data_list.append({
                "path": audio_path,
                "token": torch.tensor(
                    [
                        label_dict[x] \
                        for x in item['positive_labels'].split(',') \
                        if x in  label_dict
                    ]
                )
            })
    return token_data_list

def audioset_tokener_write(
        data_list, label_dict, data_dir, subdir, tqdm_config,
        file_name,
    ):
    temp_data_list = audioset_tokener(
        data_list, label_dict, data_dir, subdir,
        tqdm_config,
    )
    # logging.debug(json.dumps(temp_data_list[0]))
    with open(file_name, "w") as fw:
        for item in temp_data_list:
            if type(item["token"]) == torch.Tensor:
                item["token"] = item["token"].tolist()
            logging.debug(f"item: {item}")
            fw.write(f"{json.dumps(item)}\n")
    logging.info(f"write token file {file_name} done")
    return temp_data_list

def audio_mean(x):
    return torch.mean(x, dim=0).unsqueeze(0)

class AUDIO_CHANNEL_SELECT():
    def __init__(self, mono_channel):
        super().__init__()
        self.mono_channel = int(mono_channel)

    def __call__(self, x):
        return x[self.mono_channel, :].unsqueeze(0)

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
            # self.mono_func = lambda x: torch.mean(x, dim=0).unsqueeze(0)
            self.mono_func = audio_mean
        elif mono_channel.isdigit() and len(mono_channel) == 1:
            # self.mono_func = lambda x: x[int(mono_channel), :].unsqueeze(0)
            self.mono_func = AUDIO_CHANNEL_SELECT(mono_channel)
        else:
            raise ValueError(
                f"the mono_channel setting wrong, please read the help in AMINO/configs/datamodule.py"
            )

    def set_preprocesses(self, preprocesses_func):
        self.preprocess_func = preprocesses_func

    def standard_read(self, path, start=0, end=-1):
        speed = random.choices(
            self.speed_perturb,
            self.speed_perturb_weight,
            k=1,
        )[0] if self.speed_perturb is not None else 1.0
        data, fs = load_wav_with_speed_resample(
            path, speed, self.fs, start, end,
        )
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

    def dump(self, path, data_list=None):
        if data_list is None:
            data_list = self.data_list

        with open(path, "w") as fw:
            for item in data_list:
                if type(item["token"]) == torch.Tensor:
                    item["token"] = item["token"].tolist()
                fw.write(f"{json.dumps(item)}\n")

    def load(self, path):
        data_list = []
        for line in open(path, "r"):
            item = json.loads(line.strip())
            if not (("path" in item) and ("token" in item)):
                continue
            if type(item["token"]) == list:
                item["token"] = torch.tensor(item["token"])
            data_list.append(item)
        return data_list

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

    def dump(self, path, data_list=None):
        if data_list is None:
            data_list = list()
            for dataset in self.datasets:
                data_list += dataset.data_list

        with open(path, "w") as fw:
            for item in data_list:
                if type(item["token"]) == torch.Tensor:
                    item["token"] = item["token"].tolist()
                fw.write(f"{json.dumps(item)}\n")

    def load(self, path):
        list = []
        for line in open(path, "r"):
            item = json.loads(line.strip())
            if not (("path" in item) and ("token" in item)):
                continue
            if type(item["token"]) == list:
                item["token"] == torch.tensor(item["token"])
            list.append(item)
        return list

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
        files = normal_files + anormal_files
        self.data_list = []
        num_normal = len(normal_files)
        for i, file in enumerate(files):
            self.data_list.append({"file": file, "token": (i < num_normal)})

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        item = self.file_list[index]
        data = self.standard_read(item["file"])
        if data is None:
            return None
        label = torch.tensor([item["token"]])
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
        token_nj=1,
        json_path=None,
    ):
        super().__init__(fs, mono_channel, speed_perturb, speed_perturb_weight)
        json_path = hydra.utils.to_absolute_path(json_path)
        label_list = audioset_csv_reader(
            os.path.join(data_dir, "metadata", "class_labels_indices.csv"),
            ["index", "mid", "display_name"],
        )
        label_dict = audioset_label_dict_builder(label_list)
        self.num_classes = len(label_dict.values())
        if json_path is not None and os.path.isfile(json_path):
            logging.info(f"load json from {json_path}")
            self.data_list = super().load(json_path)
        else:

            audio_dir = os.path.join(data_dir, "audios")
            logging.debug(f"audioset {dataset} dataset prepare start")
            if dataset == "balanced_train":
                file_name = os.path.join(
                    data_dir, "metadata", "balanced_train_segments.csv",
                )
                segments_dir = os.path.join(audio_dir, "balanced_train_segments")
            elif dataset == "unbalanced_train":
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
                    dataset shoule be set between balanced_train/unbalanced_train/eval,
                    now it is {dataset}.
                """)
            assert os.path.isdir(segments_dir), \
                "the segment dir {segments_dir} donesn't exist"
            data_list = audioset_csv_reader(
                file_name,
                ["YTID", "start_seconds", "end_seconds", "positive_labels"],
            )
            subdir = False if dataset == "balanced_train" or dataset == "eval" else True
            if token_nj == "flexible":
                token_nj = int(mp.cpu_count())
            elif token_nj == "half_flexible":
                token_nj = int(mp.cpu_count() / 2 )
            logging.info(f"{dataset} dataset start token with {token_nj} nj")
            if token_nj > 1:
                # with open(json_path, "w") as fw:
                #     audioset_tokener_write(
                #         data_list, label_dict, segments_dir, subdir,
                #         {"position": 0, "disable": False}, fw
                #     )
                with mp.Pool(
                        processes=token_nj,
                        initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)
                    ) as pool, tempfile.TemporaryDirectory() as tempD:
                    chunked_data_list = list_split(data_list, token_nj)
                    logging.debug(f"len chunked_data_list: {len(chunked_data_list)}")
                    tasks = [
                        (
                            x, label_dict, segments_dir, subdir,
                            {"position": i},
                            os.path.join(tempD, f"{i}.scp"),
                        ) \
                        for i, x in enumerate(chunked_data_list)
                    ]
                    
                    s = pool.starmap(audioset_tokener_write, tasks)
                    logging.info(f"all token job done")
                    # self.data_list = list_merge(list(s))
                    file_merge([x[-1] for x in tasks], json_path)
                    self.data_list = super().load(json_path)
            else:
                self.data_list = audioset_tokener_write(
                    data_list, label_dict, segments_dir, subdir,
                    {"position": 0, "disable": False}, json_path
                )
        logging.info(
            f"audioset {dataset} dataset with {len(self.data_list)} item prepare done"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        logging.debug(f"Reading {index}th data")
        item = self.data_list[index]
        try:
            data = self.standard_read(item['path'])
        except Exception as e:
            logging.warning(f"{e} wrong during data {item['path']} reading, skip it")
            return None
        labels = item['token']
        label_encode = torch.zeros(self.num_classes, dtype=torch.int64)
        for label in labels:
            label_encode[label] = 1
        logging.debug(f"Reading {index}th data done")
        return data, label_encode
    
    def get_num_classes(self):
        return self.num_classes
