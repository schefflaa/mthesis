import os
import cv2
from tqdm import tqdm

for file in tqdm(os.listdir("poc/data/rorschacherberg")):

    if os.path.exists("poc/data/rsb-resized/" + file):
        continue

    try:
        img = cv2.imread("poc/data/rorschacherberg/" + file)
        new = cv2.resize(img, (250, 250))
        cv2.imwrite("poc/data/rsb-resized/" + file, new)
        
    except Exception as e:
        print("Error with file", file)
        print(e)