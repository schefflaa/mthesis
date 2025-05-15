import sys, os, cv2, torch, json, pytz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F, lightning as L
import pandas as pd, numpy as np, datetime as dt, matplotlib.pyplot as plt
from torchvision.transforms import v2
from utility.metrics.metrics import compute_metrics
from utility.metrics.statistical_metrics import compute_statistics
from torch.utils.data import DataLoader
from tqdm import tqdm
from astral import LocationInfo
from astral.sun import sun, zenith # https://sffjunkie.github.io/astral/

stgallen_city = LocationInfo('St. Gallen', 'Switzerland', 'Europe/Zurich', 47.424492554512014, 9.376722938498643)
get_sun_times = lambda date: sun(stgallen_city.observer, date, tzinfo=stgallen_city.timezone)
get_sun_zenith_deg = lambda loc, datetime: zenith(loc.observer, datetime)

class ConsistentTransform:
    # This class ensures that the same transformation is applied to each image in the sequence.
    # This is important for data augmentation techniques that rely on randomness.
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        # https://github.com/pytorch/vision/issues/9#issuecomment-789308878
        # Apply same transformation to each image by resetting the RNG
        if not imgs:
            return imgs
        rng_state = torch.get_rng_state()
        result = []
        for img in imgs:
            torch.set_rng_state(rng_state.clone())
            result.append(self.transform(img))
        return result

class pvMultipleImgSeqDataset(torch.utils.data.Dataset):
    def __init__(self, pv_file, webcams_file, img_dir, config, mode, ymin=None, ymax=None):
        self.annotations    = pd.read_csv(pv_file, sep=";", encoding="utf-8", parse_dates=["to"])
        self.webcams        = pd.read_csv(webcams_file, sep=";", encoding="utf-8")
        self.img_dir        = img_dir
        self.mode           = mode
        self.locations      = config.get('locations')
        self.num_workers    = config.get('num_workers')
        self.transform      = config.get('transform')
        self.seq_length     = config.get('seq_length')
        self.f_horizon      = config.get('f_horizon')
        self.f_resolution   = config.get('f_resolution')
        self.lead_time      = config.get('lead_time')
        self.logtransform   = config.get('logtransform')
        self.pvhistory      = config.get('pvhistory')
        self.sunpos         = config.get('sunpos')
        self.teacherforcing = config.get('teacherforcing')


        # Compute valid sequence starting points
        self.N = self.init_seq_idx()
        lookup_table = self.annotations[self.annotations["seq_idx"].notna()]["seq_idx"].to_dict() 
        self.lookup_table = {v: k for k, v in lookup_table.items()}
    
        if self.logtransform:
            self.annotations["energy-produced-Wh"] = np.log1p(self.annotations["energy-produced-Wh"])
        
        # For scaling the target values
        if self.mode == "train":
            self.transform = ConsistentTransform(self.transform)
            self.ymin = self.annotations["energy-produced-Wh"].min()
            self.ymax = self.annotations["energy-produced-Wh"].max()
        else:
            self.transform = v2.Compose([       # Default "transformation"
                v2.ToImage(),                           # Converts input to a tensor image of shape (CxHxW)
                v2.ToDtype(torch.float32, scale=True),  # Converts to float and scales to [0, 1]
            ])
            self.ymin = ymin
            self.ymax = ymax

        self.annotations["y_scaled"] = (self.annotations["energy-produced-Wh"]-self.ymin) / (self.ymax-self.ymin)

    def get_sample_weights(self, weight_fn):
        # Emphasize higher values more strongly with a power or exponential transformation
        weights = self.annotations.dropna(subset=["seq_idx"])["energy-produced-Wh"].values
        weights = weights / self.ymax  # normalize to [0, 1]
        if weight_fn:
            weights = weight_fn(weights)
        else:
            weights = [1] * len(weights)  # default weights / uniform distribution
        return torch.tensor(weights, dtype=torch.float)

    def init_seq_idx(self):
        self.annotations["to_(naive)"] = pd.to_datetime(self.annotations["to_(naive)"])
        self.annotations["seq_idx"] = pd.NA
        seq_idx = 0

        # Group by to process each day separately
        for day, group in self.annotations.groupby(self.annotations["to_(naive)"].dt.date):
            day_df = group.copy()

            # Find indices for first and last image that day
            first_image_idx = [None] * len(self.locations)
            last_image_idx = [None] * len(self.locations)
            for i, loc in enumerate(self.locations):
                first_image_idx[i] = day_df.index[day_df[f"firstImg_{loc}"]].min()
                last_image_idx[i] = day_df.index[day_df[f"lastImg_{loc}"] ].max()

            if any(pd.isna(first_image_idx)) or any(pd.isna(last_image_idx)):
                continue 

            first_common_img_idx = max(first_image_idx)
            last_common_img_idx = min(last_image_idx)
            
            # dawn is the first-1 with atday = True
            dawn_idx  = day_df.index[day_df["atday"]].min() - self.seq_length - self.lead_time + 1
            # dusk the last+1 with atday = True
            dusk_idx  = day_df.index[day_df["atday"]].max() - self.seq_length - self.lead_time + 1

            # Define the range for sequence indexing
            start_idx = max(dawn_idx, first_common_img_idx - self.seq_length)
            end_idx   = min(dusk_idx, last_common_img_idx)

            # Assign sequence indices
            for i in range(start_idx, end_idx+1):
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
        idx_first_image = self.lookup_table[seq_idx]
        idx_last_image = idx_first_image + self.seq_length -1

        images = [[] for _ in range(len(self.locations))]
        x_ts = [[] for _ in range(len(self.locations))]

        # target values
        y_ts, y_scaled = [], []
        idx_first_y = idx_last_image + self.lead_time
        idx_last_y  = idx_first_y + self.f_horizon
        for y_idx in range(idx_first_y, idx_last_y+1, self.f_resolution): # +1 to include the last y
            y_ts.append(int(self.annotations.iloc[y_idx]["to"].timestamp()))
            y_scaled.append(float(self.annotations.iloc[y_idx]["y_scaled"]))

        sunpos = [] if self.sunpos else None
        pvhistory = [] if self.pvhistory else None

        # Collect the model input (image sequence, features) and the corresponding timesteps
        for i in range(idx_first_image, idx_last_image+1): # +1 to include the last image
            for l, loc in enumerate(self.locations):
                
                # Timestep i has an image
                if self.annotations.iloc[i][f"hasImg_{loc}"]: # and self.annotations.iloc[i]["atday"]:
                    ts = int(self.annotations.iloc[i]["to"].timestamp())
                    img_path = os.path.join(self.img_dir[l], self.annotations.iloc[i]["to"].strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
                    img = cv2.imread(img_path) # the decoded images are in HWC layout (height, width, channels), where C is BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2
                
                # Timestep i has no image but is during the day.
                # -> we need to find the closest previous timestep with an image.
                elif not self.annotations.iloc[i][f"hasImg_{loc}"]:  
                    j = i-1 # Move backward until we find an image or reach nighttime
                    while not self.annotations.iloc[j][f"hasImg_{loc}"]:
                        j -= 1
                    else:  # while condition is False, we found a valid earlier image
                        if self.annotations.iloc[j][f"lastImg_{loc}"]:  # earlier image is the last image of previous day
                            img = np.zeros((250, 250, 3), dtype=np.uint8)  # Create a black image
                            ts = 0
                        else:
                            ts = int(self.annotations.iloc[j]["to"].timestamp())
                            img_path = os.path.join(self.img_dir[l], self.annotations.iloc[j]["to"].strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise Exception("Critical Error in Dataloader.") 
                
                images[l].append(img)
                x_ts[l].append(ts)
        
            # additional features
            if self.sunpos:
                sp = get_sun_zenith_deg(stgallen_city, self.annotations.iloc[i]["to"])
                sunpos.append([float(np.sin(np.radians(sp))), float(np.cos(np.radians(sp)))])
            if self.pvhistory:
                po = float((self.annotations.iloc[i]["energy-produced-Wh"]-self.ymin) / (self.ymax-self.ymin))
                pvhistory.append(po)

        if self.transform: # apply same transformation to all images in a sequence
            for loc_idx in range(len(self.locations)):
                images[loc_idx] = self.transform(images[loc_idx])

        # Tensor transforms
        y_scaled = torch.tensor(y_scaled)                                   # -> [n]
        y_ts = torch.tensor(y_ts)                                           # -> [n]
        x_ts = torch.tensor(x_ts)                                           # -> [loc, seq]
        image_sequence = torch.stack([torch.stack(x) for x in images])      # -> [loc, seq, C, H, W]
        image_sequence = image_sequence.permute(1, 0, 2, 3, 4)              # -> [seq, loc, C, H, W]

        # additional features
        features = torch.empty(self.seq_length, 0)                          # -> [seq, 0]
        if self.sunpos:
            sunpos = torch.tensor(sunpos)                                   # -> [seq, 2]  
            features = torch.cat((features, sunpos), dim=1)                 
        if self.pvhistory:
            pvhistory = torch.tensor(pvhistory).unsqueeze(-1)               # -> [seq, 1]
            features = torch.cat((features, pvhistory), dim=1)              # -> [seq, 1] if not sunpos else [seq, 2+1] 
        if self.teacherforcing:
            features = y_scaled
            
        return (image_sequence, features, y_scaled, y_ts, x_ts) 

class RoundshotMultipleDataModule(L.LightningDataModule):
    def __init__(self, data_dir, config):
        super().__init__()
        self.data_dir       = data_dir
        self.config         = config

    @property
    def min_value(self):
        return self.train_dataset.ymin

    @property
    def max_value(self):
        return self.train_dataset.ymax
    
    @property
    def logtransform(self):
        return self.config["logtransform"]

    def prepare_data(self, resizeAll=False, split=False):
        """ Prepare the data by resizing the images and splitting the target values into train, val and test sets. """

        locations = self.config["locations"]
        
        self.img_dir = [os.path.join(self.data_dir, x+"-resized") for x in locations]
        og_img_loc = [os.path.join(self.data_dir, "..", x) for x in locations]
        pv_loc = os.path.join(self.data_dir, "stg-werkhof-interpolated-total.csv")

        if resizeAll:
            for i, path in enumerate(self.img_dir):
                if os.path.exists(path):
                    print(f"Data directory {path} already present! Only resizing images not already resized.")
                else:
                    os.makedirs(path)

                # RESIZE THE ORIGINAL IMAGES
                sorted_imgfiles = os.listdir(og_img_loc[i])
                sorted_imgfiles.sort(key=lambda x: dt.datetime.strptime(x.split(".")[0], "%Y-%m-%d_%H-%M-%S"))
                for file in tqdm(sorted_imgfiles, desc=f"Resizing images for {locations[i]}"):
                    if os.path.exists(os.path.join(self.img_dir[i], file)):
                        continue # if file is already present in resized format

                    img = cv2.imread(os.path.join(og_img_loc[i], file))
                    new = cv2.resize(img, (250, 250))
                    cv2.imwrite(os.path.join(self.img_dir[i], file), new)
        
        if split:
            img_df = pd.DataFrame()
            for i, path in enumerate(self.img_dir):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Data directory ({path}) not found!")
                
                # read available imgs
                img_files = os.listdir(path)
                img_ts = [dt.datetime.strptime(img.split(".")[0], "%Y-%m-%d_%H-%M-%S") for img in img_files]
                img_ts = [pytz.timezone("Europe/Zurich").localize(ts) for ts in img_ts]
                img_files = pd.DataFrame({"filename": img_files, "timestamp": img_ts, "location": locations[i]})
                img_files.sort_values("timestamp", inplace=True)
                img_files.reset_index(drop=True, inplace=True)
                img_df = pd.concat([img_df, img_files])

            # read target
            pv = pd.read_csv(pv_loc, sep=";", parse_dates=["from", "to"], encoding="utf-8")

            # align pv and img timeframe
            # start date is the latest date of the first full image and the first pv timestamp
            img_start = img_df.groupby("location")["timestamp"].min().max()
            img_end = img_df.groupby("location")["timestamp"].max().min()

            # common interval
            img_start = max(img_start, dt.datetime(2024, 9, 19, 14, 10, 0, tzinfo=pytz.timezone("Europe/Zurich")))
            img_end   = min(img_end,   dt.datetime(2025, 3, 14, 22, 00, 0, tzinfo=pytz.timezone("Europe/Zurich")))

            start = max(img_start, pv["to"].min()).replace(hour=23, minute=50)
            end = min(img_end, pv["to"].max()).replace(hour=0, minute=10, second=0)
            
            img_df = img_df[(img_df.timestamp >= start) & (img_df.timestamp <= end)]
            pv = pv[(pv["to"] >= start) & (pv["to"] <= end)]
            pv["atday"] = pv["to"].apply(lambda x: get_sun_times(x)["dawn"] <= x <= get_sun_times(x)["dusk"])

            # add has_image column
            pv['to_(naive)'] = [x.replace(tzinfo=None) for x in pv['to']]
            for loc in locations:
                pv[f"hasImg_{loc}"] = pv["to"].isin(img_df[img_df.location == loc].timestamp)
                pv[f"firstImg_{loc}"] = pv.groupby(pv['to_(naive)'].dt.date)[f"hasImg_{loc}"].transform(lambda x: x & (x.cumsum() == 1))
                pv[f"lastImg_{loc}"] = pv.groupby(pv['to_(naive)'].dt.date)[f"hasImg_{loc}"].transform(lambda x: x & (x[::-1].cumsum() == 1))
                # add col that is true for all ts between before firstImg_ (morning) and after lastImg_ (evening the day before)
                # pv[f"truenight_{loc}"] = pv.groupby(pv['to_(naive)'].dt.date)[f"firstImg_{loc}"].transform(lambda x: x.shift(1).fillna(False))

            # split target (y) into train, val and test
            val1_start = start
            val1_end = val1_start + pd.Timedelta(days=7)

            val2_start = dt.datetime(2024, 12, 25, 0, 10, 0, tzinfo=pytz.timezone("Europe/Zurich"))
            val2_end = val2_start + pd.Timedelta(days=7)

            test_end = end
            test_start = test_end - pd.Timedelta(days=7)

            val3_end = test_start
            val3_start = val3_end - pd.Timedelta(days=7)

            test_y = pv[(pv["to"] > test_start) & (pv["to"] <= test_end)]
            val3_y = pv[(pv["to"] > val3_start) & (pv["to"] <= val3_end)]
            val2_y = pv[(pv["to"] > val2_start) & (pv["to"] <= val2_end)]
            val1_y = pv[(pv["to"] > val1_start) & (pv["to"] <= val1_end)]
            val_y = pd.concat([val1_y, val2_y, val3_y]) 
            train_y = pv[~pv["to"].isin(val_y["to"]) & ~pv["to"].isin(test_y["to"])]

            # save to csv
            train_y.to_csv(os.path.join(self.data_dir, "train_y.csv"), index=False, sep=";", encoding="utf-8")
            val_y.to_csv(os.path.join(self.data_dir, "val_y.csv"), index=False, sep=";", encoding="utf-8")
            test_y.to_csv(os.path.join(self.data_dir, "test_y.csv"), index=False, sep=";", encoding="utf-8")

    def setup(self, stage: str):
        """ Setup the data split for the given stage ('fit' or 'test'). """
        if stage == "fit":
            self.train_dataset = pvMultipleImgSeqDataset(
                pv_file      = os.path.join(self.data_dir, "train_y.csv"),
                webcams_file = os.path.join(self.data_dir, "webcams.csv"),
                img_dir      = self.img_dir,
                config  = self.config,
                mode    = "train",
            )
            
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.val_dataset = pvMultipleImgSeqDataset(
                pv_file      = os.path.join(self.data_dir, "val_y.csv"),
                webcams_file = os.path.join(self.data_dir, "webcams.csv"),
                img_dir = self.img_dir,
                config  = self.config,
                mode    = "val",
                ymin    = ymin,
                ymax    = ymax,
            )

        elif stage == "test":
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.test_dataset = pvMultipleImgSeqDataset(
                pv_file      = os.path.join(self.data_dir, "test_y.csv"), 
                webcams_file = os.path.join(self.data_dir, "webcams.csv"),
                img_dir = self.img_dir,
                config  = self.config,
                mode    = "test",
                ymin    = ymin,
                ymax    = ymax,
            )

    def train_dataloader(self):
        weights = self.train_dataset.get_sample_weights(self.config["weight_fn"])
        sampler = torch.utils.data.WeightedRandomSampler(
            weights     = weights,
            num_samples = len(weights),
            replacement = True
        )
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], sampler=sampler, num_workers=self.config["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=self.config["num_workers"])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=self.config["num_workers"])



