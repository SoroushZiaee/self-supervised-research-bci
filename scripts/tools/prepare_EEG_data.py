import os
import shutil
import argparse

import boto3
import botocore

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Download .mat files from S3
def download_patient_data(bucket_name, Prefix, verbose=True):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    path = os.path.join(os.getcwd(), "data")
    os.makedirs(path, exist_ok=True)

    for s3_file in bucket.objects.filter(Prefix=Prefix):
        file_object = s3_file.key

        object_list = file_object.split("/")

        data_name, file_name = str(object_list[1]), str(object_list[-1])

        if "." in file_name:
            if verbose:
                print(data_name, file_name)

            if verbose:
                print(path)

            path = os.path.join(os.getcwd(), "data", "mat_files", data_name)
            os.makedirs(path, exist_ok=True)

            bucket.download_file(file_object, os.path.join(path, file_name))


# covert all .mat files to .npy files
def convrt_to_npy(targetr_path, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

    mat_file_names = [
        name for name in os.listdir(targetr_path) if name.endswith(".mat")
    ]

    for mat_file_name in mat_file_names:

        try:
            data = loadmat(os.path.join(targetr_path, mat_file_name))
        except:
            continue
            print(mat_file_name)

        if not os.path.isdir(os.path.join(save_path, mat_file_name)):
            os.makedirs(os.path.join(save_path, mat_file_name), exist_ok=True)
        else:
            shutil.rmtree(os.path.join(save_path, mat_file_name))
            os.makedirs(os.path.join(save_path, mat_file_name), exist_ok=True)

        for channel_data, channel_name in zip(data["data"], data["chs"]):
            np.save(
                os.path.join(save_path, mat_file_name, f"{channel_name}.npy"),
                channel_data,
            )


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--Download", type=str, default="False")
    parser.add_argument(
        "--bucket_name", type=str, default="data-eeg-unsupervised-approach"
    )
    parser.add_argument("--prefix", type=str, default="EEG-data-processed/")
    parser.add_argument("--targetr_path", type=str, default="data/mat_files")
    parser.add_argument("--save_path", type=str, default="data/npy_files")
    opts = parser.parse_args()

    return opts


if __name__ == "__main__":

    opts = parse_args()
    conf = vars(opts)

    if conf["Download"] == "True":
        download_patient_data(conf["bucket_name"], conf["prefix"], verbose=False)

    for patien in range(1, 5):

        temp_conf = conf.copy()

        targetr_path = os.path.join(temp_conf["targetr_path"], f"P{patien}")
        save_path = os.path.join(temp_conf["save_path"], f"P{patien}")

        convrt_to_npy(targetr_path, save_path)


# !python prepare_EEG_data.py --Download True \
# 	--prefix EEG-data-processed/ \
# 	--targetr_path data/mat_files \
# 	--save_path data/npy_files
