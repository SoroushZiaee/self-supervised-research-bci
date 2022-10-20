from typing import Dict, Tuple, Union, List
import os
import random

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging


import warnings


class BCI2aDataset(Dataset):
    def __init__(
        self,
        eeg_electrode_positions: Dict[str, Tuple[int, int]],
        data_path: str,
        meta_data=None,
        transforms=None,
        is_flatten: bool = False,
    ):
        self.eeg_electrode_positions = eeg_electrode_positions
        self.data_path = data_path
        self._is_flatten = False
        self.num_channel = 22

        if meta_data is None:
            self.meta_data = pd.read_csv(os.path.join(self.data_path, "metadata.csv"))
        else:
            self.meta_data = meta_data

        self.transforms = transforms
        self.flatten_channel = []
        self.flatten_label = []

        if is_flatten:
            self.make_flatten()

    def get_ptients(self):
        return self.meta_data["patient"].values

    def get_labels(self):
        return (
            self.flatten_label if self._is_flatten else self.meta_data["label"].values
        )

    def get_sampling_rate(self):
        return 250

    def get_resampling_rate(self):
        return 256

    def get_class_distribution(self):
        return self.meta_data["label"].value_counts()

    @staticmethod
    def _shuffle(a: List, b: List):
        c = list(zip(a, b))
        random.shuffle(c)
        return zip(*c)

    def make_flatten(self):
        print("*" * 100)
        print(" - read data to memory")
        for i in range(len(self)):
            eeg_data, label = self.load_data(i)
            for j in range(self.num_channel):
                self.flatten_channel.append(eeg_data[j])
                self.flatten_label.append(label)

        self.flatten_channel, self.flatten_label = self._shuffle(
            self.flatten_channel, self.flatten_label
        )
        self._is_flatten = True

        self.eeg_electrode_positions = {
            "Cz": (0, 0),
        }
        print(" - data transfer completed")
        print("*" * 100)

    def load_data(self, idx: int):
        root_logger = logging.getLogger("mne")
        root_logger.setLevel(logging.ERROR)
        mne.set_log_level(verbose="ERROR")
        warnings.simplefilter("ignore")

        meta_data = self.meta_data.iloc[idx]
        eeg_data = np.load(
            os.path.join(self.data_path, "train" + "/" + meta_data["file_name"])
        )

        info = mne.create_info(
            list(self.eeg_electrode_positions.keys()),
            sfreq=self.get_sampling_rate(),
            ch_types="eeg",
            verbose=0,
        )

        eeg_data = (
            mne.io.RawArray(eeg_data, info, verbose=0)
            .filter(l_freq=2, h_freq=None, verbose=0)
            .resample(self.get_resampling_rate(), verbose=0)
        )

        eeg_data = eeg_data.get_data()

        label = meta_data["label"] - 1

        return eeg_data, label

    def __len__(self) -> int:
        return (
            len(self.flatten_channel)
            if self._is_flatten
            else len(self.meta_data["file_name"])
        )

    def __getitem__(self, idx: int) -> Union[dict, torch.Tensor]:

        eeg_data, label = (
            (self.flatten_channel[idx], self.flatten_label[idx])
            if self._is_flatten
            else self.load_data(idx)
        )

        wav = (
            {
                key: np.expand_dims(eeg_data, axis=0)
                for key in self.eeg_electrode_positions.keys()
            }
            if self._is_flatten
            else {
                key: np.expand_dims(eeg_data[i], axis=0)
                for i, key in enumerate(self.eeg_electrode_positions.keys())
            }
        )

        if self.transforms is not None:
            wav, label = self.transforms(wav, label)

        return wav, label

    def subset(self, indices):
        return self.__class__(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            meta_data=self.meta_data.iloc[indices],
            transforms=self.transforms,
            # Just for test
            is_flatten=True,
        )

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = torch.vstack([item[1] for item in batch]).squeeze()

        return [imgs, trgts]
