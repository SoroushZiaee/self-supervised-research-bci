from pase_eeg.lit_modules.pase_lit import PASE

import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    # For specifying on WANDB
    parser.add_argument("--data_path", type=str)

    return parser.parse_args()


def run(data_path: str):
    meta_data = pd.read_csv(os.path.join(data_path, "metadata.csv"))
    print(meta_data.head())


def main(conf):
    data_path = conf["data_path"]
    run(data_path)


if __name__ == "__main__":
    conf = vars(parse_args())
    main(conf)
