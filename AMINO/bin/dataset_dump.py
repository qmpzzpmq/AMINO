import argparser
import sys

from omegaconf import OmegaConf

from AMINO.utils.init_object import init_object

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str)
    parser.add_argument("--dataset_name", type=str, choices=["train", "val", "test"])
    parser.add_argument("--dataset_num", type=int)
    parser.add_argument("--dump_path", type=str)
    parser.add_argument("--dump_method", type=str, choices=["json", "tfrecord"])
    # parser.add_argument("--dataset", type=str, default="AMINO.datamodule.datasets:AUDIOSET_DATASET")
    return parser

def main(cmd_args):
    parser = get_parser()
    args = parser.parse_args(cmd_args)
    configs = OmegaConf.load(args.configs)

    dataset = init_object(
        configs["datamodule"]["datasets"][args.dataset_name][args.dataset_num]
    )
    dataset.dump(args.dump_path, format=args.dump_method)

if __name__ == "__main__":
    main(sys.argv[1:])