import boto3
import numpy as np
import pandas as pd
from io import StringIO
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TARGET_BUCKET = "masc12-roundshot-img-storage"
OUTLIER_CSV = "image-size-outliers.csv"

# session = boto3.Session(profile_name='exxeta-admin')
# bucket = session.resource('s3').Bucket(TARGET_BUCKET)
# s3_client = session.client('s3')

s3_client = boto3.client('s3')
bucket = boto3.resource('s3').Bucket(TARGET_BUCKET)

def get_files_and_folders_from_s3():
    # get all files in the bucket but not folders
    files = [obj.key for obj in bucket.objects.all() if obj.key.endswith('0-00.jpg')]
    files = [f for f in files if not f.startswith('test/')]
    folders = [obj.key[:-1] for obj in bucket.objects.all() if obj.key.endswith('/')]
    folders.remove('test')
    folders.remove('stg-power')
    return files, folders

def get_filesizes(files: list) -> list:
    filesizes = []
    for file in files:
        response = s3_client.head_object(Bucket=TARGET_BUCKET, Key=file)
        filesizes.append(response['ContentLength'])
    return filesizes

def organize_files_and_sizes_into_folders(files: list, sizes: list, folders: list) -> dict:
    filesizes = dict(zip(files, sizes))
    folder_filesizes = {folder: [] for folder in folders}
    for file, size in filesizes.items():
        folder = file.split('/')[0]
        folder_filesizes[folder].append(size)
    return folder_filesizes

def get_lower_outlier_threshold(filesizes: list) -> float:
    q1 = np.percentile(filesizes, 25)
    q3 = np.percentile(filesizes, 75)
    iqr = q3 - q1
    threshold = q1 - 2 * iqr
    if threshold < 10_000:
        return 10_000
    else:
        return threshold

def overwrite_csv_to_s3(lower_outlier_thresholds:pd.DataFrame) -> None:
    logger.info(f"Overwriting {OUTLIER_CSV}.")
    csv_buffer = StringIO()
    lower_outlier_thresholds.to_csv(csv_buffer, index=False, sep=';', encoding='utf-8')
    s3_client.put_object(Bucket=TARGET_BUCKET, Key=OUTLIER_CSV, Body=csv_buffer.getvalue())

def lambda_handler(event, context):
    files, folders = get_files_and_folders_from_s3()
    sizes = get_filesizes(files)
    folder_filesizes = organize_files_and_sizes_into_folders(files, sizes, folders)
    lower_outlier_thresholds = {
        folder: get_lower_outlier_threshold(sizes)
        for folder, sizes in folder_filesizes.items()
    }
    
    lower_outlier_thresholds = pd.DataFrame(lower_outlier_thresholds.items(), columns=['folder', 'lower_outlier_threshold'])
    overwrite_csv_to_s3(lower_outlier_thresholds)

    return {
        'statusCode': 200,
        'body': 'Updated lower outlier thresholds for all folders'
    }