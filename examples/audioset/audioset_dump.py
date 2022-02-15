import sys
import os
import argparse
import logging

from omegaconf import OmegaConf
from AMINO.datamodule.datamodule import AMINODataModule

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("dump_dir", type=str)
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
    if not os.path.isdir(args.dump_dir):
        os.makedirs(args.dump_dir)

    cfg = OmegaConf.load(args.config)
    logging.info(OmegaConf.to_yaml(cfg))
    datamodule = AMINODataModule(cfg['datamodule'])
    datamodule.setup()
    for datasetname in [ "val", "train", "test"]:
        logging.info(f"check datset {datasetname}")
        if (dataset := datamodule.get_dataset(datasetname)) is not None:
            file_path = os.path.join(args.dump_dir, f"{datasetname}.scp")
            if os.path.isfile(file_path):
                logging.warning(f"the file {file_path} exists, remove")
                os.remove(file_path)
                # continue
            logging.info(f"dumping dataset {datasetname} into {file_path}")
            dataset.dump(file_path)
            logging.info(f"dumping dataset {datasetname} into {file_path}, done")
        else:
            logging.warning(f"there is no {datasetname} dataset")

if __name__ == "__main__":
    main(sys.argv[1:])