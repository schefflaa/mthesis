import sys, torch, os, cv2, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poc.dataloader import RoundshotDataModule

class pvSingleImgDataset(torch.utils.data.Dataset):
    def __init__(self, pv_file, img_dir, transform=None, lead_time=1, ymin=None, ymax=None, mode=None):
        self.annotations = pd.read_csv(pv_file, sep=";", encoding="utf-8", parse_dates=["to"])
        self.annotations = self.annotations[self.annotations["has_image"] == True]
        self.img_dir = img_dir
        self.transform = transform
        self.lead_time = lead_time

        # For scaling the target values
        if mode == "train":
            self.ymin = self.annotations["energy-produced-Wh"].min()
            self.ymax = self.annotations["energy-produced-Wh"].max()
        else:
            self.ymin = ymin
            self.ymax = ymax


    def __len__(self):
        # Adjust the length to ensure there is enough data for the forecast horizon
        return len(self.annotations) - self.lead_time


    def __getitem__(self, idx):
        if not self.annotations.iloc[idx]["has_image"]:
            raise ValueError("No image found for the given index.")
        
        # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2
        # "In the case of color images, the decoded images will have the channels stored in BGR order."
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["to"].strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
        img = cv2.imread(img_path) 
        
        if self.transform:
            img = self.transform(img)

        y_idx = idx + self.lead_time  # idx of target value after 'lead_time' steps
        ts = int(self.annotations.iloc[y_idx]["to"].timestamp()) # timestamp of the target value
        y_scaled = (self.annotations.iloc[y_idx]["energy-produced-Wh"]-self.ymin) / (self.ymax-self.ymin)
        y_scaled = torch.tensor(float(y_scaled))
            
        return (img, y_scaled, ts)


class SingleImageRoundshotDataModule(RoundshotDataModule):
    def setup(self, stage: str):
        """ Setup the data split for the given stage ('fit' or 'test'). """
        if stage == "fit":
            self.train_dataset = pvSingleImgDataset(os.path.join(self.data_dir, "train_y.csv"), self.img_dir, self.transform, self.lead_time, mode="train")
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.val_dataset = pvSingleImgDataset(os.path.join(self.data_dir, "val_y.csv"), self.img_dir, self.transform, self.lead_time, ymin, ymax, mode="val")

        elif stage == "test":
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.test_dataset = pvSingleImgDataset(os.path.join(self.data_dir, "test_y.csv"), self.img_dir, self.transform, self.forecalead_timet_horizon, ymin, ymax, mode="test")