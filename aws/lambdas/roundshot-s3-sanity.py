import boto3, pytz, json
import pandas as pd
import datetime as dt
from io import StringIO
import logging, random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TARGET_BUCKET = 'masc12-roundshot-img-storage'
WEBCAM_CSV = "webcams-2024-10-14-v2.csv"
OUTLIER_CSV = "image-size-outliers.csv"

# s3_client = boto3.client('s3')
# lambda_client = boto3.client('lambda')
# bucket = boto3.resource('s3').Bucket(TARGET_BUCKET)

session = boto3.Session(profile_name='exxeta-admin')
bucket = session.resource('s3').Bucket(TARGET_BUCKET)
lambda_client = session.client('lambda')
s3_client = session.client('s3')

def get_files_and_folders_from_s3():
    # get all files in the bucket but not folders
    files = [obj.key for obj in bucket.objects.all() if obj.key.endswith('0-00.jpg')]
    files = [f for f in files if not f.startswith('test/')]
    folders = [obj.key[:-1] for obj in bucket.objects.all() if obj.key.endswith('/')]
    folders.remove('test')
    folders.remove('scheitlinsbuchel')
    folders.remove('stg-power')
    return files, folders

def get_date_range():
    now = get_nearest_10_minute()
    start = now - dt.timedelta(days=2)
    start = start.replace(hour=6, minute=0, second=0, microsecond=0)

    daterange = []
    while start <= now:
        if 7 <= start.hour <= 21:
            daterange.append(start)
        start += dt.timedelta(minutes=10)
    return daterange

def get_nearest_10_minute() -> dt.datetime:
    now = dt.datetime.now(pytz.timezone('Europe/Zurich'))
    now = now - dt.timedelta(minutes=now.minute % 10,
                             seconds=now.second,
                             microseconds=now.microsecond)
    return now

def filter_relevant_files(allfiles, daterange):
    tz = pytz.timezone('Europe/Zurich')
    start_date = daterange[0]
    return [
        f for f in allfiles
        if tz.localize(dt.datetime.strptime(f.split('/')[-1][:-4], '%Y-%m-%d_%H-%M-%S')) >= start_date
    ]


def get_csv_from_s3(bucket:str, csv:str) -> pd.DataFrame:
    csv_object = s3_client.get_object(Bucket=bucket, Key=csv)
    csv_content = csv_object['Body'].read().decode('utf-8')
    webcams_df = pd.read_csv(StringIO(csv_content), sep=';', encoding='utf-8')
    return webcams_df

def invokeChildLambda(cam_id:str, cam_name:str, urldate:str, urltime:str, min_filesize:float) -> None:
    inputParams = {
        'cam_id': cam_id,
        'cam_name': cam_name,
        'urldate': urldate,
        'urltime': urltime,
        'min_filesize': min_filesize,
        'wait_seconds': random.uniform(0, 30) # Random wait time between 0 and 60 seconds
    } 
 
    lambda_client.invoke(
        FunctionName = 'arn:aws:lambda:eu-north-1:454075690551:function:roundshot-img-fetcher',
        InvocationType = 'Event',
        Payload = json.dumps(inputParams)
    )

def fetch_missing_files(files:list, folders:list, daterange:list, webcams:pd.DataFrame, outliers:pd.DataFrame) -> None:
    missing = []
    for d in daterange:
        for fol in folders:
            if not webcams[webcams["short"] == fol].scrape.values[0]:
                continue # skip if scrape is disabled
            
            target = f"{fol}/{d:%Y-%m-%d_%H-%M-%S}.jpg"
            if target not in files: 
                logger.info(f"Missing file {target}. Invoking child lambda.")
                urldate = f"{d:%Y-%m-%d}"
                urltime = f"{d:%H-%M-%S}"
                cam_id = webcams[webcams["short"] == fol]["id"].values[0]
                min_filesize = outliers[outliers["folder"] == fol]["lower_outlier_threshold"].values[0]
                # invokeChildLambda(cam_id, fol, urldate, urltime, min_filesize)
                missing.append(target)

def lambda_handler(event, context):
    outliers = get_csv_from_s3(TARGET_BUCKET, OUTLIER_CSV)
    webcams = get_csv_from_s3(TARGET_BUCKET, WEBCAM_CSV)
    allfiles, folders = get_files_and_folders_from_s3()
    daterange = get_date_range()
    files = filter_relevant_files(allfiles, daterange)
    fetch_missing_files(files, folders, daterange, webcams, outliers)
        
    return {
        'statusCode': 200,
        'body': 'Hello from Lambda!'
    }

lambda_handler(None, None)