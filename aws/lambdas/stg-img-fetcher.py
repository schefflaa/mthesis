import requests, boto3, pytz
from typing import Tuple
import datetime as dt
import logging, time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

timeout_sec = 5
TARGET_BUCKET = 'masc12-roundshot-img-storage'

# s3_client = boto3.client('s3')
# logtable = boto3.resource('dynamodb').Table(LOG_TABLE_NAME)

session = boto3.Session(profile_name='exxeta-admin')
s3_client = session.client('s3')

def does_file_exist(foldername: str, filename: str) -> bool:	
    try:
        s3_client.head_object(Bucket=TARGET_BUCKET, Key=foldername + "/" + filename)
        return True
    except:
        return False

def save_image_to_s3(foldername: str, filename: str, img: bytes) -> None:
    s3_client.put_object(
        Bucket = TARGET_BUCKET,
        Key = foldername + "/" + filename,
        Body = img,
        ContentType = 'image/jpg'
    )
    
def perform_get_request(url: str, min_filesize: float, retries=1) -> Tuple[bytes, dict]:
    try:
        logger.info(f'GET: {url}')
        response = requests.get(url, timeout=timeout_sec)
        last_modified = response.headers.get('last-modified')

        if response.status_code != 200:
            raise Exception(f"Status code of request was {response.status_code}")
    
        if len(response.content) < min_filesize:
            raise Exception(f"File size of {len(response.content)/1024/1024:.2f} mb is smaller than the min. filesize ({min_filesize/1024/1024:.2f} mb).")
        
        if not last_modified:
            raise Exception(f"No last-modified header in response.")
        
        return response.content, {"statusCode": response.status_code, "url": url, "message":"Successfully downloaded image.", "last_modified": last_modified}
    
    except Exception as e1:
        if retries > 3:
            return None, {"statusCode": 500, "url": url, "message":"Failed to download image."}
        
        logger.error(f'💥 An error occurred: {e1}')
        time.sleep(60) # wait 1 min
        return perform_get_request(url, min_filesize, retries+1)

def get_image(cam_name: str, min_filesize:float):
        # GET REQUEST TO ROUNDSHOT STORAGE
        url = f'https://webcam.ostschweiz.ch/~webcam007/image-now.jpg'
        img, msg = perform_get_request(url, min_filesize)
        
        # STORE IMAGE IN S3
        if img:
            last_modified = msg['last_modified']
            tmp = dt.datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
            tmp = pytz.timezone('GMT').localize(tmp).astimezone(pytz.timezone('Europe/Zurich'))
            filename = f'{tmp.strftime("%Y-%m-%d_%H-%M-%S")}.jpg'

            if does_file_exist(cam_name, filename):
                logger.info(f'Image already exists in s3.')
                msg['message'] = 'Image already exists in s3.'
                return msg

            logger.info(f'Saving image to s3 ...')
            save_image_to_s3(cam_name, filename, img)
            logger.info(f'Saved ✅')
        
        return msg

def lambda_handler(event, context):
    cam_name = "scheitlinsbuchel"
    min_filesize = 10_000 # 10kb TODO adjust?
    logger.info(f"🎯 Getting image from {cam_name}.")
    info = get_image(cam_name, min_filesize)
    return info

info = lambda_handler(None, None)
print(info)