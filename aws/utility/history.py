import json, boto3, time, pytz, random
import pandas as pd
import datetime as dt
from io import StringIO
from tqdm import tqdm


BUCKET = "masc12-roundshot-img-storage"
WEBCAM_CSV = "webcams-2024-10-14-v2.csv"
OUTLIER_CSV = "image-size-outliers.csv"

session = boto3.Session(profile_name='exxeta-admin', region_name='eu-north-1')
lambda_client = session.client('lambda')
s3_client = session.client('s3')

def invokeChildLambda(cam_id:str, cam_name:str, urldate:str, urltime:str, min_filesize:float, max_wait:int=1) -> None:
    inputParams = {
      'cam_id': cam_id,
      'cam_name': cam_name,
      'urldate': urldate,
      'urltime': urltime,
      'min_filesize': min_filesize,
      'wait_seconds': random.uniform(0, max_wait) 
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

def get_date_range_from(start, end):
    try:
        start = pytz.timezone('Europe/Zurich').localize(start)
    except ValueError:
        pass

    try:
        end = pytz.timezone('Europe/Zurich').localize(end)
    except ValueError:
        pass
    
    daterange = []
    while start <= end:
        if dt.time(4, 0) <= start.time() < dt.time(23, 0):
            daterange.append(start)
        start += dt.timedelta(minutes=10)
    return daterange

def get_files_from_s3():
    bucket = session.resource('s3').Bucket("masc12-roundshot-img-storage")
    files = [obj.key for obj in bucket.objects.all() if obj.key.endswith('.jpg')]
    return files

files = get_files_from_s3()

start = dt.datetime(2024, 2, 23, 4, 0)
end = dt.datetime(2024, 9, 20, 22, 50)
history = get_date_range_from(start, end)

# for rorschacherberg
dayends = {
    2: dt.time(18, 0),
    3: dt.time(19, 30),
    4: dt.time(20, 30),
    5: dt.time(21, 0),
    6: dt.time(21, 20),
    7: dt.time(21, 20),
    8: dt.time(21, 0),
    9: dt.time(20, 0),
}

daystarts ={
    2: dt.time(8, 0),
    3: dt.time(7, 0),
    4: dt.time(6, 30),
    5: dt.time(6, 20),
    6: dt.time(6, 20),
    7: dt.time(6, 10),
    8: dt.time(6, 30),
    9: dt.time(7, 30),
}

new = ""
missing = []
tz = pytz.timezone('Europe/Zurich')

with tqdm(total=len(history)) as pbar:
    for day in history:
        pbar.update(1)

        if day.time() < daystarts[day.month] or day.time() > dayends[day.month]:
            continue

        if tz.localize(dt.datetime(2024, 4, 5, 10, 30)) <= day <= tz.localize(dt.datetime(2024, 4, 15, 17, 50)):
            continue
        
        there_are_missings = False
        get = f"{day:%Y-%m-%d_%H-%M-%S}.jpg"
        urldate = f"{day:%Y-%m-%d}"
        urltime = f"{day:%H-%M-%S}"
        
        if urldate != new:
            new = urldate

        target = f"rorschacherberg/{get}"
        if target not in files:
            missing.append(target)
            invokeChildLambda("53a838bc15a356.87772279", "rorschacherberg", urldate, urltime, 677_847.0, 30)
            there_are_missings = True

        if day >= tz.localize(dt.datetime(2024, 8, 9, 5, 0)):
            target = f"kronberg/{get}"
            if target not in files:
                missing.append(target)
                invokeChildLambda("53a838bc15a356.87772279", "kronberg", urldate, urltime, 12_524.0, 30)
                there_are_missings = True
                
        if day >= tz.localize(dt.datetime(2024, 9, 3, 5, 0)):
            target = f"wiler-turm/{get}"
            if target not in files:
                missing.append(target)
                invokeChildLambda("545b7dcf9cedf1.10734711", "wiler-turm", urldate, urltime, 40_127.25, 30)
                there_are_missings = True

        if day >= tz.localize(dt.datetime(2024, 6, 13, 5, 0)):
            target = f"golfclub-erlen/{get}"
            if target not in files:
                missing.append(target)
                invokeChildLambda("608fc16698a1e7.09133428", "golfclub-erlen", urldate, urltime, 80_621.0, 30)
                there_are_missings = True

        if day >= tz.localize(dt.datetime(2024, 9, 8, 5, 0)):
            target = f"romanshorn/{get}"
            if target not in files:
                missing.append(target)
                invokeChildLambda("62863f9fcbb216.97456372", "romanshorn", urldate, urltime, 532_205.5, 30)
                there_are_missings = True

        if day >= tz.localize(dt.datetime(2024, 9, 9, 5, 0)):
            target = f"altenrhein/{get}"
            if target not in files:
                missing.append(target)
                invokeChildLambda("5c8b6fec9dad68.39944594", "altenrhein", urldate, urltime, 561_358.0, 30)
                there_are_missings = True

            target = f"elbenalp/{get}"
            if target not in files:
                missing.append(target)
                invokeChildLambda("5a7c518a176f53.54105137", "elbenalp", urldate, urltime, 518_066.0, 30)
                there_are_missings = True        
        
        if there_are_missings: # if all images are present, skip to next day without waiting
            time.sleep(7.5)

with open("MissingFiles.txt", "w") as f:
    for line in missing:
        f.write(line + "\n")