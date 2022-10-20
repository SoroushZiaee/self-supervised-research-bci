import boto3
import os
import argparse


def download_s3_folder(s3, bucket_name, s3_folder, local_dir=None):

    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """

    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = (
            obj.key
            if local_dir is None
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        )
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)


def main():
    s3 = boto3.resource("s3")
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket_name", required=True, type=str, help="bucker name")
    parser.add_argument(
        "--s3_folder", required=True, type=str, help="folder to download from s3"
    )
    parser.add_argument("--local_dir", required=True, type=str, help="Folder to save")

    args = parser.parse_args()

    download_s3_folder(s3, args.bucket_name, args.s3_folder, args.local_dir)


if __name__ == "__main__":
    main()

# !python3 S3Downloader.py --bucket_name "data-eeg-unsupervised-approach" --s3_folder "EEG-data-processed" --local_dir "/data/Klinik_dataset"
