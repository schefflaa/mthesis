import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os, scienceplots, cv2, tqdm, sys

from skimage import io

webcams = ['romanshorn', 'altenrhein', 'kronberg', 'wiler-turm', 'golfclub-erlen', 'sitterdorf', 'ebenalp', 'rorschacherberg']
webcam_indicator = []
brightnesses = []
timesteps = []
filesizes = []

errors = []

for wbc in tqdm.tqdm(webcams):

    wbc = "ebenalp" if wbc == "elbenalp" else wbc

    for file in tqdm.tqdm(os.listdir(f'data/{wbc}'), leave=False):

        ts = dt.datetime.strptime(file, "%Y-%m-%d_%H-%M-%S.jpg")
        size = int(os.path.getsize(f'data/{wbc}/{file}'))
        
        try:
            location = f'data/{wbc}/{file}'
            
            with open(location, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                raise Exception('Not complete image: check_chars != b"\xff\xd9"')
            else:
                _ = io.imread(location)
                img = cv2.imread(location)

            # brightness = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
            brightness = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()

        except Exception as e: 
            errors.append(f"{location} \t {e}")
            print(location, e)
        
        brightnesses.append(brightness)
        webcam_indicator.append(wbc)
        timesteps.append(ts)
        filesizes.append(size)

df_timesteps = pd.DataFrame({"webcam": webcam_indicator, "timestamp": timesteps, "brightness": brightnesses, "filesize": filesizes})	
df_timesteps.to_csv("image-intensity-df.csv", sep=";", encoding="utf-8", index=False)
with open("errors.txt", "w") as f:
    for error in errors:
        f.write(error + "\n")