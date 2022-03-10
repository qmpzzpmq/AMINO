import argparse
import sys
import logging

from omegaconf import OmegaConf

from AMINO.utils.init_object import init_object

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True)
    parser.add_argument(
        "--dataset_name", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument("--dataset_num", default=0, type=int)
    parser.add_argument("--dump_path", required=True, type=str)
    parser.add_argument(
        "--dump_method", type=str, default="tfrecord", choices=["json", "tfrecord"],
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["ALL", "INFO", "DEBUG", "WARN"],
        default="INFO",
    )
    return parser

def main(cmd_args):
    parser = get_parser()
    args = parser.parse_args(cmd_args)
    logging_level = eval("logging." + args.logging_level)
    logging.basicConfig(
        level=logging_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info(args)
    configs = OmegaConf.load(args.configs)

    dataset_conf = configs["datamodule"]["datasets"][
        args.dataset_name
    ][
        args.dataset_num
    ]
    logging.info(
        f"preparing dataset {dataset_conf} into {args.dump_path} with method {args.dump_method}"
    )
    dataset = init_object(dataset_conf)
    dataset.prepare_data()
    dataset.setup()

    logging.info(f"dump into {args.dump_method}")
    dataset.dump(args.dump_path, format=args.dump_method)

if __name__ == "__main__":
    main(sys.argv[1:])
