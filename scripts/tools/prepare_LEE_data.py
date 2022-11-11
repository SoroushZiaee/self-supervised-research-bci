import os
import re
import glob
import urllib.request
import argparse

import numpy as np
import pandas as pd
from scipy.io import loadmat


class LEE_matFile:
    def __init__(self, data_path):
        self.mat_file = loadmat(data_path, squeeze_me=True, struct_as_record=False)

    def get_train_signal(self):
        return self.mat_file["EEG_MI_train"].smt.transpose((1, 2, 0))

    def get_train_classes(self):
        return self.mat_file["EEG_MI_train"].y_class

    def get_test_signal(self):
        return self.mat_file["EEG_MI_test"].smt.transpose((1, 2, 0))

    def get_test_classes(self):
        return self.mat_file["EEG_MI_test"].y_class

    def get_channels(self):
        return self.mat_file["EEG_MI_train"].chan

    def get_sample_rate(self):
        return self.mat_file["EEG_MI_train"].fs

    def __str__(self):
        return (
            "sample rate : "
            + str(self.get_sample_rate())
            + "\n"
            + "channel list : "
            + str(self.get_channels())
            + "\n"
            + "number of trains : "
            + str(self.get_train_signal().shape[0])
            + "\n"
            + "number of tests : "
            + str(self.get_test_signal().shape[0])
        )


def download_lee_dataset(targetr_path, session):

    for _ in range(1, 53):
        urllib.request.urlretrieve(
            "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/session{}/s{}/sess{:02d}_subj{:02d}_EEG_MI.mat".format(
                session, _, session, _
            ),
            filename=os.path.join(
                targetr_path, "sess{:02d}_subj{:02d}_EEG_MI.mat".format(session, _)
            ),
        )

        print("sess{:02d}_subj{:02d}_EEG_MI.mat".format(session, _))


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--targetr_path", type=str, default="data/mat_files")
    parser.add_argument("--session", type=str, default="1")
    parser.add_argument("--save_path", type=str, default="data/npy_files")
    opts = parser.parse_args()

    return opts


if __name__ == "__main__":

    opts = parse_args()
    conf = vars(opts)

    download_lee_dataset(conf["targetr_path"], int(conf["session"]))

    file_pathes = []
    label = []
    data_type = []
    patient_id = []
    for _ in glob.glob(os.path.join(conf["targetr_path"], "*.mat")):

        lee = LEE_matFile(_)
        label.extend(lee.get_train_classes())
        label.extend(lee.get_test_classes())

        for idx, signal in enumerate(lee.get_train_signal()):
            np.save(
                os.path.join(
                    conf["save_path"],
                    os.path.split(_)[1].replace(".mat", "")
                    + f"_train_signal_trial_{idx}.npy",
                ),
                signal,
            )

            file_pathes.append(
                os.path.split(_)[1].replace(".mat", "")
                + f"_train_signal_trial_{idx}.npy"
            )
            data_type.append("train")
            patient_id.append(
                int(re.search(r"\d{1,2}", os.path.basename(_).split("_")[1]).group())
            )

        for idx, signal in enumerate(lee.get_test_signal()):
            np.save(
                os.path.join(
                    conf["save_path"],
                    os.path.split(_)[1].replace(".mat", "")
                    + f"_test_signal_trial_{idx}.npy",
                ),
                signal,
            )

            file_pathes.append(
                os.path.split(_)[1].replace(".mat", "")
                + f"_test_signal_trial_{idx}.npy"
            )
            data_type.append("test")
            patient_id.append(
                int(re.search(r"\d{1,2}", os.path.basename(_).split("_")[1]).group())
            )

        print(os.path.split(_)[1])

    data = {
        "file_pathes": file_pathes,
        "label": label,
        "data_type": data_type,
        "patient_id": patient_id,
    }

    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(conf["save_path"], "meta_data.csv"), index=False)
