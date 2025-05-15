import sys, os, cv2, torch, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F, lightning as L
import pandas as pd, numpy as np, datetime as dt, matplotlib.pyplot as plt
from utility.metrics.metrics import compute_metrics
from utility.metrics.statistical_metrics import compute_statistics
from torch.utils.data import DataLoader
from tqdm import tqdm

class pvImgSeqDataset(torch.utils.data.Dataset):
    def __init__(self, pv_file, img_dir, transform=None, seq_length=5, lead_time=1, ymin=None, ymax=None, mode=None):
        self.annotations = pd.read_csv(pv_file, sep=";", encoding="utf-8", parse_dates=["to"])
        self.img_dir = img_dir
        self.transform = transform
        self.seq_length = seq_length
        self.lead_time = lead_time

        # Compute valid sequence starting points
        self.N = self.init_seq_idx()
        lookup_table = self.annotations[self.annotations["seq_idx"].notna()]["seq_idx"].to_dict() 
        self.lookup_table = {v: k for k, v in lookup_table.items()}
    
        # For scaling the target values
        if mode == "train":
            self.ymin = self.annotations["energy-produced-Wh"].min()
            self.ymax = self.annotations["energy-produced-Wh"].max()
        else:
            self.ymin = ymin
            self.ymax = ymax

    def init_seq_idx(self):
        self.annotations["from"] = pd.to_datetime(self.annotations["from"])
        self.annotations["to"] = pd.to_datetime(self.annotations["to"])
        self.annotations["seq_idx"] = pd.NA
        seq_idx = 0
        # Group by 'atday' to process each day separately
        for day, group in self.annotations.groupby(self.annotations["from"].dt.date):
            day_df = group.copy()

            # Find indices for first and last image that day
            first_image_idx = day_df.index[day_df["is_first_with_image_that_day"]].min()
            last_image_idx = day_df.index[day_df["is_last_with_image_that_day"]].max()

            if pd.isna(first_image_idx) or pd.isna(last_image_idx):
                continue  # Skip if no valid sequence for the day

            # Define the range for sequence indexing
            start_idx = max(0, first_image_idx - self.seq_length + 1)  # Ensure we don't go out of bounds
            end_idx = last_image_idx + 1  # Include the last image

            # Assign sequence indices
            for i in range(start_idx, end_idx):
                self.annotations.at[i, "seq_idx"] = seq_idx
                seq_idx += 1

        # Convert seq_idx to integer
        self.annotations["seq_idx"] = pd.to_numeric(self.annotations["seq_idx"], errors="coerce").astype("Int64")
        return seq_idx # return the total amount of sequences 

    def __len__(self):
        # Adjust the length to ensure there is enough data for the forecast horizon
        # return len(self.annotations) - self.seq_length - self.lead_time + 1
        return self.N

    def __getitem__(self, seq_idx):
        """ Get the image sequence and the target value for the given index. """
        images = []
        timesteps = []
        start_idx = self.lookup_table[seq_idx]
        end_idx = start_idx + self.seq_length 
        
        # target value
        y_idx = (end_idx-1) + self.lead_time  # idx of target value after 'lead_time' steps
        y_ts = int(self.annotations.iloc[y_idx]["to"].timestamp()) # timestamp of the target value
        y_scaled = (self.annotations.iloc[y_idx]["energy-produced-Wh"]-self.ymin) / (self.ymax-self.ymin)
        y_scaled = torch.tensor(float(y_scaled))

        # model input 
        for i in range(start_idx, end_idx): # end_idx is exclusive
            if self.annotations.iloc[i]["has_image"]: # has image
                # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2
                ts = int(self.annotations.iloc[i]["to"].timestamp())
                img_path = os.path.join(self.img_dir, self.annotations.iloc[i]["to"].strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
                img = cv2.imread(img_path) # the decoded images are in HWC layout (height, width, channels), where C is BGR
            
            elif self.annotations.iloc[i]["atday"]: # no image + is at day
                j = i - 1 # Finding the closest previous image
                while not self.annotations.iloc[j]["has_image"]:
                    j -= 1 
                ts = int(self.annotations.iloc[j]["to"].timestamp())
                img_path = os.path.join(self.img_dir, self.annotations.iloc[j]["to"].strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
                img = cv2.imread(img_path)
            
            else: # no image + is night
                img = np.zeros((250, 250, 3), dtype = np.uint8)  # Create a black image
                ts = 0

            if self.transform:
                # If ToTensor() is used in the transform, the image is converted to a torch.FloatTensor of shape (CxHxW) w/ range [0.0, 1.0]:
                img = self.transform(img)
            
            images.append(img)
            timesteps.append(ts)
        image_sequence = torch.stack(images)
        timesteps = torch.tensor(timesteps)
        return (image_sequence, y_scaled, y_ts, timesteps)

class RoundshotDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=4, num_workers=0, transform=None, seq_length=5, lead_time=1, shuffle_train=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.seq_length = seq_length
        self.lead_time = lead_time
        self.shuffle_train = shuffle_train
    
    @property
    def min_value(self):
        return self.train_dataset.ymin

    @property
    def max_value(self):
        return self.train_dataset.ymax

    def prepare_data(self, resizeAll=False, split=False):
        """ Prepare the data by resizing the images and splitting the target values into train, val and test sets. """
        
        self.img_dir = os.path.join(self.data_dir, "rsb-resized")
        og_img_loc = os.path.join(self.data_dir, ".." ,"rorschacherberg")
        pv_loc = os.path.join(self.data_dir, "stg-werkhof-interpolated-total.csv")

        if resizeAll:
            if os.path.exists(self.img_dir):
                print(f"Data directory {self.img_dir} already present! Only resizing images not already resized.")
            else:
                os.makedirs(self.img_dir)

            # RESIZE THE ORIGINAL IMAGES
            for file in tqdm(os.listdir(og_img_loc)):
                newfilepath = os.path.join(self.img_dir, file)

                if os.path.exists(newfilepath):
                        continue # if file is already present in resized format

                img = cv2.imread(os.path.join(og_img_loc, file))
                new = cv2.resize(img, (250, 250))
                cv2.imwrite(newfilepath, new)
        
        if split:
            if not os.path.exists(self.img_dir):
                raise FileNotFoundError(f"Data directory ({self.img_dir}) not found!")

            # read available imgs
            img_files = os.listdir(self.img_dir)
            img_ts = [dt.datetime.strptime(img.split(".")[0], "%Y-%m-%d_%H-%M-%S") for img in img_files]
            img_files = pd.DataFrame({"filename": img_files, "timestamp": img_ts})
            img_files.sort_values("timestamp", inplace=True)
            img_files.reset_index(drop=True, inplace=True)

            # read target
            pv = pd.read_csv(pv_loc, sep=";", parse_dates=["from", "to"], encoding="utf-8")

            img_start = img_files.timestamp.min()
            img_end = img_files.timestamp.max()

            # TODO CHANGE (common interval)
            # img_start = dt.datetime(2024, 7, 19, 14, 10, 0) #, tzinfo=pytz.timezone("Europe/Zurich"))
            # img_end = dt.datetime(2025, 2, 4, 13, 40, 0) # , tzinfo=pytz.timezone("Europe/Zurich"))

            # align pv and img timeframe
            start = max(img_start, pv["to"].min())
            end = min(img_end, pv["to"].max())

            img_files = img_files[(img_files.timestamp >= start) & (img_files.timestamp <= end)]
            pv = pv[(pv["to"] >= start) & (pv["to"] <= end)]
            
            # add has_image column
            pv["has_image"] = pv["to"].apply(lambda x: x.to_pydatetime() in img_ts)

            # add is_day column
            pv['is_first_with_image_that_day'] = pv.groupby(pv['from'].dt.date)['has_image'].transform(lambda x: x & (x.cumsum() == 1))
            pv['is_last_with_image_that_day'] = pv.groupby(pv['from'].dt.date)['has_image'].transform(lambda x: x & (x[::-1].cumsum() == 1))
            for i, row in pv.iterrows():
                if row["is_first_with_image_that_day"]:
                    atday = True

                if atday:
                    pv.at[i, "atday"] = True
                else:
                    pv.at[i, "atday"] = False

                if row["is_last_with_image_that_day"]:
                    atday = False

            # split target (y) into train, val and test
            test_size_days, val_size_days = 7, 7
            train_y = pv[pv["to"] <= end - pd.Timedelta(days=test_size_days+val_size_days)]
            val_y = pv[(pv["to"] > end - pd.Timedelta(days=test_size_days+val_size_days)) & (pv["to"] <= end - pd.Timedelta(days=test_size_days))]
            test_y = pv[pv["to"] > end - pd.Timedelta(days=test_size_days)]

            # save to csv
            train_y.to_csv(os.path.join(self.data_dir, "train_y.csv"), index=False, sep=";", encoding="utf-8")
            val_y.to_csv(os.path.join(self.data_dir, "val_y.csv"), index=False, sep=";", encoding="utf-8")
            test_y.to_csv(os.path.join(self.data_dir, "test_y.csv"), index=False, sep=";", encoding="utf-8")

    def setup(self, stage: str):
        """ Setup the data split for the given stage ('fit' or 'test'). """
        if stage == "fit":
            self.train_dataset = pvImgSeqDataset(os.path.join(self.data_dir, "train_y.csv"), self.img_dir, self.transform, self.seq_length, self.lead_time, mode="train")
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.val_dataset = pvImgSeqDataset(os.path.join(self.data_dir, "val_y.csv"), self.img_dir, self.transform, self.seq_length, self.lead_time, ymin, ymax, mode="val")

        elif stage == "test":
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.test_dataset = pvImgSeqDataset(os.path.join(self.data_dir, "test_y.csv"), self.img_dir, self.transform, self.seq_length, self.lead_time, ymin, ymax, mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

