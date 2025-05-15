import json, boto3, time, pytz
import pandas as pd
import datetime as dt
from io import StringIO
from boto3.dynamodb.conditions import Key
import logging, random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

LOG_TABLE_NAME = "masc12-roundshot-log"
BUCKET = "masc12-roundshot-img-storage"

WEBCAM_CSV = "webcams-2024-10-14-v2.csv"
OUTLIER_CSV = "image-size-outliers.csv"

# logtable = boto3.resource('dynamodb').Table(LOG_TABLE_NAME)
# lambda_client = boto3.client('lambda')
# s3_client = boto3.client('s3')

session = boto3.Session(profile_name='exxeta-admin')
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

def wait_for_responses(childKeys: list) -> list[dict]:
    results = []
    waiting_for_responses = True
    while waiting_for_responses:
        logger.info("Waiting for responses...")
        for ck in childKeys:
            response = logtable.query(KeyConditionExpression=Key('datetime_camera').eq(ck))
            results += response['Items']

            if len(results) == len(childKeys): # Check if all children have reported back
                waiting_for_responses = False
                break
            else:
                time.sleep(1) 
    return results

def lambda_handler(event, context):
    now = get_nearest_10_minute()

    if now.hour < 7 or now.hour > 22:
        logger.info("Outside of working hours. Exiting.")
        return
    
    urldate = f"{now:%Y-%m-%d}"
    urltime = f"{now:%H-%M-%S}"
    webcams = get_csv_from_s3(BUCKET, WEBCAM_CSV)
    outliers = get_csv_from_s3(BUCKET, OUTLIER_CSV)
    childKeys = []

    for _, row in webcams.iterrows():
        if not row["scrape"]:
            continue

        cam_id = row["id"]
        cam_name = row["short"]
        childKeys.append(f"{urldate}_{urltime}_{cam_name}")
        min_filesize = outliers[outliers["folder"] == cam_name]["lower_outlier_threshold"].values[0]
        # invokeChildLambda(cam_id, cam_name, urldate, urltime, min_filesize)
        print(f"Starting scraper for {cam_name} ({cam_id}) with min_filesize {min_filesize/1024/1024:.2f} MB.")
    
    return {
        'statusCode': 200,
        'body': f'Started all ({len(childKeys)}) scrapers.'
    }


lambda_handler(None, None)