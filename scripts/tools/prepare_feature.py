from pase_eeg.lit_modules.pase_lit import PASE
from pase_eeg.lit_modules.utils import eeg_electrode_configs

from pase_eeg.data.bci.bci_dataset import BCI2aDataset
from pase_eeg.data.transforms import Compose, ToTensor

import argparse
import os
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # For specifying on WANDB
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--electrode_path", type=str)
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--emb_dim", type=int)

    return parser.parse_args()


def prepare_dataset(data_path: str, eeg_electrode_positions, transforms):
    dataset = BCI2aDataset(eeg_electrode_positions, data_path, transforms)
    dataset.make_flatten()

    return dataset


def prepare_model(electrode_path, emb_dim: int, pretrained_backend_weights_path: str):
    pase = PASE(
        electrode_path,
        emb_dim,
        pretrained_backend_weights_path=pretrained_backend_weights_path,
    )

    pase.to("cuda:0")

    return pase


def run(data_path: str, electrode_path: str, weight_path: str, emb_dim: int):
    (
        eeg_electrode_positions,
        eeg_electrods_plane_shape,
    ) = eeg_electrode_configs(electrode_path)

    transforms = Compose([ToTensor(device="cuda:0")])

    dataset = prepare_dataset(data_path, eeg_electrode_positions, transforms)
    model = prepare_model(electrode_path, emb_dim, weight_path)

    wav, label = dataset[0]
    wav = 
    print(wav.keys())
    print(wav["Cz"].size)
    print(type(wav["Cz"]))

    embeddings = model.forward(wav)


def main(conf):
    data_path = conf["data_path"]
    electrode_path = conf["electrode_path"]
    weight_path = conf["weight_path"]
    emb_dim = conf["emb_dim"]
    run(data_path, electrode_path, weight_path, emb_dim)


if __name__ == "__main__":
    conf = vars(parse_args())
    main(conf)
