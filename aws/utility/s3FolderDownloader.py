import boto3, os
from tqdm import tqdm

session = boto3.Session(profile_name='exxeta-admin')
s3 = session.resource('s3')

def download_s3_folder(bucket_name, s3_folder, local_dir, localfiles):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    
    bucket = s3.Bucket(bucket_name)
    count = 0
    for obj in bucket.objects.filter(Prefix=s3_folder):
        count += 1
    
    with tqdm(total=count) as pbar:
        for obj in tqdm(bucket.objects.filter(Prefix=s3_folder)):
            pbar.update(1)

            if obj.key[-1] == '/':
                continue

            if obj.key.split("/")[1] in localfiles: # check if file is already downloaded
                continue

            target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            
            bucket.download_file(obj.key, target)

s3bucket     =  'masc12-roundshot-img-storage'
s3folders    = ['altenrhein', 'elbenalp', 'golfclub-erlen', 'kronberg', 'romanshorn', 'rorschacherberg', 'sitterdorf', 'wiler-turm']
localfolders = ['altenrhein', 'ebenalp', 'golfclub-erlen', 'kronberg', 'romanshorn', 'rorschacherberg', 'sitterdorf', 'wiler-turm']

for i in range(len(s3folders)):
    s3_folder = s3folders[i]
    target = f"data/{localfolders[i]}"
    localfiles = os.listdir(target)

    print(f"\nNumber of files in local folder {target}: ", len(localfiles))
    print(f"Downloading {s3_folder} to {target}...")
    download_s3_folder(s3bucket, s3_folder, target, localfiles)
    print("Number of files in local folder: ", len(os.listdir(target)))

# run from root dir:
# >>> mthesis_env/bin/python aws/utility/s3FolderDownloader.py