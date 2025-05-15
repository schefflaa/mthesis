import json, boto3, pytz
import pandas as pd
import datetime as dt
from io import StringIO
from boto3.dynamodb.conditions import Key
import logging, random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

LOG_TABLE_NAME = "masc12-roundshot-log"
TARGET_BUCKET = "masc12-roundshot-img-storage"
WEBCAM_CSV = "webcams-2024-10-14-v2.csv"
OUTLIER_CSV = "image-size-outliers.csv"

# logtable = boto3.resource("dynamodb").Table(LOG_TABLE_NAME)
# lambda_client = boto3.client('lambda')
# s3_client = boto3.client('s3')

session = boto3.Session(profile_name='exxeta-admin', region_name='eu-north-1')
logtable = session.resource('dynamodb').Table(LOG_TABLE_NAME)
lambda_client = session.client('lambda')
s3_client = session.client('s3')

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

def get_csv_from_s3(bucket:str, csv:str) -> pd.DataFrame:
    csv_object = s3_client.get_object(Bucket=bucket, Key=csv)
    csv_content = csv_object['Body'].read().decode('utf-8')
    webcams_df = pd.read_csv(StringIO(csv_content), sep=';', encoding='utf-8')
    return webcams_df

def get_nearest_10_minute() -> dt.datetime:
    now = dt.datetime.now(pytz.timezone('Europe/Zurich'))
    now = now - dt.timedelta(minutes=now.minute % 10,
                             seconds=now.second,
                             microseconds=now.microsecond)
    return now

def get_failed_requests():
    now = get_nearest_10_minute()
    twentyfour_hours_ago = now - dt.timedelta(hours=24)

    response = logtable.scan(
        FilterExpression=Key('statusCode').eq(418) & Key('datetime_camera').between(twentyfour_hours_ago.strftime("%Y-%m-%d_%H-%M-%S"), now.strftime("%Y-%m-%d_%H-%M-%S"))
    )

    return response['Items']

def lambda_handler(event, context):
    childKeys = []
    webcams = get_csv_from_s3(TARGET_BUCKET, WEBCAM_CSV)
    outliers = get_csv_from_s3(TARGET_BUCKET, OUTLIER_CSV)
    failed_requests = get_failed_requests()
    
    logger.info(f"There are {len(failed_requests)} failed requests in the past 24 hours.")
    
    for item in failed_requests:
        urldate, urltime, cam_name = item["datetime_camera"].split("_")
        
        if not webcams[webcams["short"] == cam_name].scrape.values[0]:
            continue # skip if scrape is disabled
        
        childKeys.append(item["datetime_camera"])
        cam_id = webcams[webcams["short"] == cam_name]["id"].values[0]
        min_filesize = outliers[outliers["folder"] == cam_name]["lower_outlier_threshold"].values[0]
        # invokeChildLambda(cam_id, cam_name, urldate, urltime, min_filesize)
        logger.info(f"Invoked child lambda for {cam_name} with min_filesize {min_filesize/1024/1024:.1f} MB.")
    
    return {
        'statusCode': 200,
        'body': json.dumps('Started all backuppers')
    }

lambda_handler(None, None)