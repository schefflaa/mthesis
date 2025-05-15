import requests, boto3
from utility import InfoMessage
from typing import Tuple
import logging, time, re

logger = logging.getLogger()
logger.setLevel(logging.INFO)

timeout_sec = 5
TARGET_BUCKET = 'masc12-roundshot-img-storage'

# s3_client = boto3.client('s3')

session = boto3.Session(profile_name='exxeta-admin', region_name='eu-north-1')
s3_client = session.client('s3')

def get_csv(min_filesize:float) -> InfoMessage:

        url = f'https://daten.stadt.sg.ch/api/explore/v2.1/catalog/datasets/aktuelle-stromproduktion-der-solaranlagen-der-stgaller-stadtwerke/exports/csv?lang=de&timezone=Europe/Zurich&use_labels=true&delimiter=;'

        # GET REQUEST TO STG
        csv, msg, when = perform_get_request(url, min_filesize)
        filename = f'{when}.csv'
        foldername = f'stg-power'

        # CHECK IF FILE EXISTS ALREADY
        if does_file_exist(foldername, filename):
            logger.info(f'File already exists: {foldername}/{filename}')
            return InfoMessage(200, url, 'File already exists') 
        
        # STORE CSV IN S3
        if csv:
            logger.info(f'Saving csv to s3 ...')
            save_csv_to_s3(foldername, filename, csv)
            logger.info(f'Saved ✅')
        
        return msg

def check_response(response: requests.Response, min_filesize: float) -> None:
    if response.status_code != 200:
        raise Exception(f"Status code of request was {response.status_code}")
    
    if len(response.content) < min_filesize:
        raise Exception(f"File size of {len(response.content)/1024/1024:.2f} mb is smaller than the min. filesize ({min_filesize/1024/1024:.2f} mb).")

def perform_get_request(url: str, min_filesize: float, retries:int=1) -> Tuple[bytes, InfoMessage, str]:
    try:
        logger.info(f'🎯 GET: {url}')
        response = requests.get(url, timeout=timeout_sec)
        check_response(response, min_filesize)

        # Extract the last modified date
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?([+-]\d{2}:\d{2}|Z)"
        when = re.search(pattern, response.text).group(0)
        timestamp = when.replace('T', '_').replace('+02:00', '').replace('+01:00', '').replace(':', '-')

        if not timestamp:
            raise Exception('Could not extract timestamp from csv.')

        return response.content, InfoMessage(response.status_code, url, 'Successfully downloaded csv.'), timestamp
    
    except Exception as e1:
        if retries > 3:
            return None, InfoMessage(418, url, f'Could not download csv: {e1}')

        logger.error(f'💥 An error occurred: {e1}.')
        logger.error(f'... Retrying ...')
        time.sleep(60)
        return perform_get_request(url, min_filesize, retries+1)

def save_csv_to_s3(foldername: str, filename: str, csv: bytes) -> None:
    s3_client.put_object(
        Bucket = TARGET_BUCKET,
        Key = foldername + "/" + filename,
        Body = csv,
        ContentType = 'text/csv'
    )
    
def does_file_exist(foldername: str, filename: str) -> bool:	
    try:
        s3_client.head_object(Bucket=TARGET_BUCKET, Key=foldername + "/" + filename)
        return True
    except:
        return False

def lambda_handler(event, context):
    logger.info(f"Getting csv...")
    res = get_csv(1_000) # 1kb
    return res.to_dict()

res = lambda_handler(None, None)
print(res)