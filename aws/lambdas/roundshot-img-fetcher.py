import requests, boto3, pytz
from utility import InfoMessage
from typing import Tuple
import datetime as dt
import logging, time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

timeout_sec = 5
TARGET_BUCKET = 'masc12-roundshot-img-storage'
LOG_TABLE_NAME = 'masc12-roundshot-log'

# s3_client = boto3.client('s3')
# logtable = boto3.resource('dynamodb').Table(LOG_TABLE_NAME)

session = boto3.Session(profile_name='exxeta-admin')
s3_client = session.client('s3')
logtable = session.resource('dynamodb').Table(LOG_TABLE_NAME)

def get_image(cam_id: str, cam_name: str, urldate:str, urltime: str, min_filesize:float, wait:float=0) -> InfoMessage:
        """
        Function to download a single image from a specific camera at a specific date and time.

        Parameters:
        ----------
        cam_id : str
            Camera ID as defined in the 'webcams.csv' file.
        
        cam_name : str
            Camera name (short) as defined in the 'webcams.csv' file.

        urldatetime : datetime
            Date and time.

        min_filesize : float
            Minimum file size in bytes. If the file size is smaller than this value, the function will return an error.
        """

        filename = f'{urldate}_{urltime}.jpg'
        foldername = f'{cam_name}'

        url1 = f'https://storage.roundshot.com/{cam_id}/{urldate}/{urltime}/{urldate}-{urltime}_full.jpg'
        url2 = f'https://storage2.roundshot.com/{cam_id}/{urldate}/{urltime}/{urldate}-{urltime}_full.jpg'
        url3 = f'https://archive.roundshot.com/{cam_id}/{urldate}/{urltime}/{urldate}-{urltime}_full.jpg'
        url4 = f'https://archive1.roundshot.com/{cam_id}/{urldate}/{urltime}/{urldate}-{urltime}_full.jpg'
        url5 = f'https://archive2.roundshot.com/{cam_id}/{urldate}/{urltime}/{urldate}-{urltime}_full.jpg'
        url6 = f'https://archive3.roundshot.com/{cam_id}/{urldate}/{urltime}/{urldate}-{urltime}_full.jpg'

        # CHECK IF FILE EXISTS ALREADY
        if does_file_exist(foldername, filename):
            logger.info(f'File already exists: {foldername}/{filename}')
            return InfoMessage(200, url1, 'File already exists') 
        
        # WAIT
        time.sleep(wait)
        
        # GET REQUEST TO ROUNDSHOT STORAGE
        img, msg = perform_get_request(url1, url2, url3, url4, url5, url6, min_filesize)
        
        # STORE IMAGE IN S3
        if img:
            logger.info(f'Saving image to s3 ...')
            save_image_to_s3(foldername, filename, img)
            logger.info(f'Saved ✅')
        
        return msg

def check_response(response: requests.Response, min_filesize: float) -> None:
    if response.status_code != 200:
        raise Exception(f"Status code of request was {response.status_code}")
    
    if len(response.content) < min_filesize:
        raise Exception(f"File size of {len(response.content)/1024/1024:.2f} mb is smaller than the min. filesize ({min_filesize/1024/1024:.2f} mb).")

def perform_get_request(url1: str, url2: str, url3: str, url4: str, url5: str, url6: str, min_filesize: float, try_number:int=1) -> Tuple[bytes, InfoMessage]:
    try:
        logger.info(f'🎯 GET: {url1}')
        response = requests.get(url1, timeout=timeout_sec)
        check_response(response, min_filesize)
        return response.content, InfoMessage(response.status_code, url1, 'Successfully downloaded image.')
    
    except Exception as e1:
        try:
            logger.warning(f'⚠️ First attempt ("storage") failed: {e1}')
            logger.info(f'🎯 GET: {url2}')
            response = requests.get(url2, timeout=timeout_sec)
            check_response(response, min_filesize)
            return response.content, InfoMessage(response.status_code, url2, 'Successfully downloaded image.')
                
        except Exception as e2:
            try:
                logger.warning(f'⚠️ Second attempt ("storage2") failed: {e2}')
                logger.info(f'🎯 GET: {url3}')
                response = requests.get(url3, timeout=timeout_sec)
                check_response(response, min_filesize)
                return response.content, InfoMessage(response.status_code, url3, 'Successfully downloaded image.')
            
            except Exception as e3:
                try:
                    logger.warning(f'⚠️ Third attempt ("archive") failed: {e3}')
                    logger.info(f'🎯 GET: {url4}')
                    response = requests.get(url4, timeout=timeout_sec)
                    check_response(response, min_filesize)
                    return response.content, InfoMessage(response.status_code, url4, 'Successfully downloaded image.')
                
                except Exception as e4:
                    try:
                        logger.warning(f'⚠️ Fourth attempt ("archive1") failed: {e4}')
                        logger.info(f'🎯 GET: {url5}')
                        response = requests.get(url5, timeout=timeout_sec)
                        check_response(response, min_filesize)
                        return response.content, InfoMessage(response.status_code, url5, 'Successfully downloaded image.')
                    except Exception as e5:
                        try:
                            logger.warning(f'⚠️ Fifth attempt ("archive2") failed: {e4}')
                            logger.info(f'🎯 GET: {url6}')
                            response = requests.get(url6, timeout=timeout_sec)
                            check_response(response, min_filesize)
                            return response.content, InfoMessage(response.status_code, url6, 'Successfully downloaded image.')

                        except Exception as e6:
                            logger.warning(f'⚠️ Sixth attempt ("archive3") failed: {e4}')
                            
                            if try_number >= 3:
                                logger.error(f'❌ All attempts failed. Giving up after try number {try_number}.')
                                return None, InfoMessage(418, url4, f'Failed to download image. e1: {e1}, e2: {e2}, e3: {e3}, e4: {e4}, e5: {e5}, e6: {e6}')
                            
                            logger.info(f'🔁 Retrying all after {60*try_number} seconds ...')
                            time.sleep(60*try_number)
                            return perform_get_request(url1, url2, url3, url4, url5, url6, min_filesize, try_number+1)


def save_image_to_s3(foldername: str, filename: str, img: bytes) -> None:
    s3_client.put_object(
        Bucket = TARGET_BUCKET,
        Key = foldername + "/" + filename,
        Body = img,
        ContentType = 'image/jpg'
    )
    
def log_result_to_dynamodb(result: InfoMessage, cam_name: str, urldate: str, urltime: str) -> None:
    if result.statusCode == 204: 
        return # img already exists
    
    now = dt.datetime.now(pytz.timezone('Europe/Zurich'))
    when = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    key = f"{urldate}_{urltime}_{cam_name}"
    
    logtable.put_item(Item={
        "datetime_camera": key,
        "timestamp": when,
        "statusCode": result.statusCode,
        "message": result.message,
        "url": result.url,
        "ttl": int((now + dt.timedelta(days=3)).timestamp())
    })

def does_file_exist(foldername: str, filename: str) -> bool:	
    try:
        s3_client.head_object(Bucket=TARGET_BUCKET, Key=foldername + "/" + filename)
        return True
    except:
        return False

def lambda_handler(event, context):
    cam_id = event["cam_id"]
    cam_name = event["cam_name"]
    urldate = event["urldate"]
    urltime = event["urltime"]
    min_filesize = event["min_filesize"]
    wait = event["wait_seconds"]
    
    logger.info(f"Getting image from {cam_name}.")
    res = get_image(cam_id, cam_name, urldate, urltime, min_filesize, wait)
    logger.info(f"Logging to dynamodb: {res}")
    log_result_to_dynamodb(res, cam_name, urldate, urltime)
    return res.to_dict()

res = lambda_handler({
    "cam_id": "61373586c8a715.78521672",
    "cam_name": "test",
    "urldate": "2024-10-14",
    "urltime": "15-40-00",
    "min_filesize": 300_000.0, # 300kb
    "wait_seconds": 1.7
}, None)

print(res)