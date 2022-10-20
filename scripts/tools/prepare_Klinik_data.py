import numpy as np
import pandas as pd
from scipy.io import loadmat

import os
import argparse
from typing import (
    List,
)


class ClinikData(object):
    def __init__(self, filePath: str):

        self._matFile = loadmat(filePath)

    def get_signal(self) -> np.ndarray:
        return self._matFile["data"]

    def get_n_data_points(self) -> int:
        """
        Number of data points
        """
        return self._matFile["data"].shape[1]

    def get_channels_name(self) -> list:
        """
        Names of channels
        """
        return self._matFile["chs"]

    def get_sampling_rate(self) -> float:
        """
        Get the frequency
        """
        return self._matFile["fs"][0][0]

    def get_file_duration(self) -> float:
        """
        Returns the file duration in seconds
        """

        return self.get_n_data_points() / self.get_sampling_rate()

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"number data points    : {self.get_n_data_points()}\n"
            + f"# file duration       : {self.get_file_duration()}\n"
            + f"# channels name       : {self.get_channels_name()}\n"
            + f"# sampling rate       : {self.get_sampling_rate()}\n"
            + "#############################################################\n"
        )


class PrepareKlinikData:
    def __init__(
        self,
        target_path: str,
        save_path: str,
        patient_list: List[int],
        length=1,
    ):

        self.target_path = target_path
        self.save_path = save_path
        self.patient_list = patient_list
        self.length = length

    def _mat_meta_data(self):
        for i, patient_num in enumerate(self.patient_list):
            if not i:
                self.mat_meta_data = self._extract_patients_meta_data(patient_num)
            else:
                self.mat_meta_data = pd.concat(
                    (
                        self.mat_meta_data,
                        self._extract_patients_meta_data(patient_num),
                    ),
                    axis=0,
                )

        self.mat_meta_data.reset_index(drop=True, inplace=True)

    def _extract_filenames(self, data_path):
        out_dict = {"meta_file": [], "eeg": []}

        for item in os.listdir(data_path):
            if ".xlsx" in item:
                out_dict["meta_file"].append(item)
            elif ".mat" in item:
                out_dict["eeg"].append(item)

        return out_dict

    def _extract_start_end_seizure(self, meta_data):
        start_seizures = []
        end_seizures = []
        for i, row in enumerate(meta_data.iterrows()):
            if row[1]["EventString"] == "Seizure start":
                start_seizures.append(row[1]["StartSecond"])

            elif row[1]["EventString"] == "Seizure end":
                end_seizures.append(row[1]["StartSecond"])

        return start_seizures, end_seizures

    def _extract_patients_meta_data(self, patient_number):

        path = os.path.join(self.target_path, f"P{patient_number}")
        dict_files = self._extract_filenames(path)
        patient_ids = [item.split("_")[3] for item in dict_files["meta_file"]]

        patient_meta_data_template = "export_events_Exported_{}_20211103.xlsx"

        dict_out = {
            "patient_number": [],
            "record_id": [],
            "file_path": [],
            "label": [],
            "start_seizures": [],
            "end_seizures": [],
        }
        for i, patient_id in enumerate(patient_ids):
            meta_path = os.path.join(
                path, patient_meta_data_template.format(patient_id)
            )
            meta = pd.read_excel(meta_path, engine="openpyxl")

            start_seizures, end_seizures = self._extract_start_end_seizure(meta)
            dict_out["patient_number"].append(patient_number)
            dict_out["record_id"].append(patient_id)
            dict_out["file_path"].append(
                meta_path.replace(
                    "export_events_Exported", "export_eeg_full_Exported"
                ).replace(".xlsx", ".mat")
            )
            if len(start_seizures) == 0 or len(end_seizures) == 0:
                dict_out["start_seizures"].extend([0])
                dict_out["end_seizures"].extend([0])
                dict_out["label"].extend([1])
            else:
                dict_out["start_seizures"].extend(start_seizures)
                dict_out["end_seizures"].extend(end_seizures)
                dict_out["label"].extend([0])

        df = pd.DataFrame(dict_out)
        return df

    def get_class_distribution(self):
        return self.npy_meta_data["label"].value_counts()

    def __call__(self):
        self._mat_meta_data()

        file_pathe = []
        label = []

        for idx in range(len(self.mat_meta_data)):

            meta_data = self.mat_meta_data.iloc[idx]

            try:
                clinik_data = ClinikData(meta_data["file_path"])
                signal = clinik_data.get_signal()
            except:
                continue

            for _ in range(
                int(clinik_data.get_file_duration() // self.length) - self.length
            ):

                if _ == int(meta_data["start_seizures"]) or _ == int(
                    meta_data["end_seizures"]
                ):
                    continue
                else:
                    save_path = os.path.join(
                        self.save_path,
                        os.path.split(meta_data["file_path"])[1].replace(".mat", "")
                        + f"_trial_{_}.npy",
                    )

                    np.save(
                        save_path,
                        signal[
                            :,
                            _
                            * self.length
                            * clinik_data.get_sampling_rate() : (_ + 1)
                            * self.length
                            * clinik_data.get_sampling_rate(),
                        ],
                    )

                    file_pathe.append(save_path)

                    if (
                        _ > meta_data["start_seizures"]
                        and _ < meta_data["end_seizures"]
                    ):
                        label.append(1)
                    else:
                        label.append(0)

        meta_dataframe = {
            "file_pathe": file_pathe,
            "label": label,
        }

        self.npy_meta_data = pd.DataFrame(data=meta_dataframe)
        self.npy_meta_data.to_csv(
            os.path.join(self.save_path, "meta_data.csv"), index=False
        )
        print(self.get_class_distribution())


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, default="data/mat_files")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--length", type=int)
    opts = parser.parse_args()

    return opts


if __name__ == "__main__":

    opts = parse_args()
    conf = vars(opts)

    PrepareKlinikData(
        target_path=conf["target_path"],
        save_path=conf["save_path"],
        patient_list=[1, 2, 3, 4],
        length=conf["length"],
    ).__call__()
