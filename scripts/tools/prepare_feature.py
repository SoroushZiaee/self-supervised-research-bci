from pase_eeg.lit_modules.pase_lit import PASE
from pase_eeg.lit_modules.utils import eeg_electrode_configs

from pase_eeg.data.bci.bci_dataset import BCI2aDataset
from pase_eeg.data.transforms import Compose, ToTensor

from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import os
import pandas as pd
import numpy as np

import torch
from torch import Tensor


def parse_args():
    parser = argparse.ArgumentParser()
    # For specifying on WANDB
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--electrode_path", type=str)
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--output_path", type=str)

    return parser.parse_args()


def prepare_dataset(data_path: str, eeg_electrode_positions, transforms):
    dataset = BCI2aDataset(eeg_electrode_positions, data_path, transforms=transforms)
    dataset.make_flatten()

    return dataset


def preprocess_data(x: Dict[str, Tensor]):

    x["Cz"] = torch.squeeze(x["Cz"])
    x = torch.permute(
        torch.vstack(list(map(lambda a: a.unsqueeze(0), x.values()))),
        (1, 0, 3, 2),
    )

    # x.to("cuda:0")

    return x


def prepare_model(electrode_path, emb_dim: int, pretrained_backend_weights_path: str):
    pase = PASE(
        electrode_path,
        emb_dim,
        pretrained_backend_weights_path=pretrained_backend_weights_path,
    )

    # pase.to("cuda:0")

    return pase


def save_features(x, y, output_path):
    np.save(os.path.join(output_path, "X.npy"), x.cpu().detach().numpy())
    np.save(os.path.join(output_path, "y.npy"), y)


def run(
    data_path: str,
    electrode_path: str,
    weight_path: str,
    emb_dim: int,
    output_path: str,
):
    (
        eeg_electrode_positions,
        eeg_electrods_plane_shape,
    ) = eeg_electrode_configs(electrode_path)

    transforms = Compose([ToTensor(device="cuda:0")])

    dataset = prepare_dataset(data_path, eeg_electrode_positions, transforms)
    model = prepare_model(electrode_path, emb_dim, weight_path)

    wav, label = dataset[:]
    wav = preprocess_data(wav[:, :, :100, :])

    embeddings = model.forward(wav)

    labels = np.array([l["label"].item() for l in label])

    print(f"embedding shape : {embeddings.size()}")
    print(f"embedding shape : {labels.shape}")

    save_features(embeddings, labels, output_path)


def main(conf):
    data_path = conf["data_path"]
    electrode_path = conf["electrode_path"]
    weight_path = conf["weight_path"]
    emb_dim = conf["emb_dim"]
    output_path = conf["output_path"]
    run(data_path, electrode_path, weight_path, emb_dim, output_path)


if __name__ == "__main__":
    conf = vars(parse_args())
    main(conf)
