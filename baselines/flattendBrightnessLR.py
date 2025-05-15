import sys, torch, os, cv2, datetime as dt, pandas as pd, numpy as np
import lightning as L, torch.nn as nn, torchvision.transforms as transforms
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import SingleImageRoundshotDataModule, pvSingleImgDataset
from models.LitBaseModel import LitBaseModel

# --------------------------------- Arguments ------------------------------------
execution_time = dt.datetime.now()
if sys.argv[1] == "debug":
    RUN_NAME = "[debug] " + execution_time.strftime("%Y-%m-%d_%H-%M-%S")
    IS_TRAIN = False
elif sys.argv[1] == "train":
    IS_TRAIN = True
    RUN_NAME = execution_time.strftime("%Y-%m-%d_%H-%M-%S")
else:
    raise ValueError("Invalid argument @ position 1. Use 'debug' or 'train'.")

IS_VERBOSE = (sys.argv[2] == "verbose" if len(sys.argv) > 2 else False)


# --------------------------------- Model definition ------------------------------------

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        """
        input_size: size of the input image (assuming square image)
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear((input_size**2), 1)  # Single linear layer

    def forward(self, x):
        xflat = x.view(x.size(0), -1)  # Flatten the image
        return self.linear(xflat)

class LitLRModel(LitBaseModel):
    def initialize_model(self, **kwargs):
        return LinearRegressionModel(**kwargs)

class gray_pvSingleImgDataset(pvSingleImgDataset):
    def __getitem__(self, idx):
        if not self.annotations.iloc[idx]["has_image"]:
            raise ValueError("No image found for the given index.")
        
        # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2
        # "In the case of color images, the decoded images will have the channels stored in BGR order."
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["to"].strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
        img = cv2.imread(img_path)
        
        # Convert to HSV (Hue, Saturation, Value) color space 
        # Then, only take the Value channel (brightness)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

        if not IS_TRAIN:
            cv2.imwrite(f"viz/grayFulloriginal.png", img)    
        
        if self.transform:
            img = self.transform(img)

        _, y_scaled, ts = super().__getitem__(idx)
        return (img, y_scaled, ts)
        
    
class gray_SingleImageRoundshotDataModule(SingleImageRoundshotDataModule):
    def setup(self, stage: str):
        """ Setup the data split for the given stage ('fit' or 'test'). """
        if stage == "fit":
            self.train_dataset = gray_pvSingleImgDataset(os.path.join(self.data_dir, "train_y.csv"), self.img_dir, self.transform, self.lead_time, mode="train")
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.val_dataset = gray_pvSingleImgDataset(os.path.join(self.data_dir, "val_y.csv"), self.img_dir, self.transform, self.lead_time, ymin, ymax, mode="val")

        elif stage == "test":
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.test_dataset = gray_pvSingleImgDataset(os.path.join(self.data_dir, "test_y.csv"), self.img_dir, self.transform, self.lead_time, ymin, ymax, mode="test")

# ------------------------------ Hyperparameters ---------------------------------
model_hp = {
    "lr": 0.001,
    "optimizer": torch.optim.Adagrad,
    "verbose": IS_VERBOSE,
    "loss_outlier_threshold": 4e5 if IS_VERBOSE else sys.float_info.max,
    "input_size": 250,
}

other_hp = {
    "batch_size":       8,                          # number of sequences in a batch
    "lead_time": 1,                          # number of steps (10 minute) ahead we want to predict
    "num_workers":      15 if IS_TRAIN else 0,      # number of workers for dataloader
    "transform":        transforms.ToTensor(),  	# converts a image/np.ndarray (HxWxC) in the range [0, 255] to a torch.FloatTensor of shape (CxHxW) w/ range [0.0, 1.0]
    "shuffle_train":    True,                       # shuffle the training data (shuffles sequences; not images within a sequence)
}

train_hp = {
    "max_epochs": 15, 
}

# -------------------------------- Setup logging ---------------------------------
print("RUN NAME: ", RUN_NAME)
logger = TensorBoardLogger(save_dir="baselines/tblogs", name="FlattendBrightnessLR", version=RUN_NAME, default_hp_metric=False)
hparams = {
    k: str(v) if not isinstance(v, (int, float, bool, tuple, list, dict, set)) else v
    for k, v in {**model_hp, **other_hp, **train_hp}.items()
}
logger.log_hyperparams(hparams, {"hp/val_rmse": 0, "hp/val_mae": 0})
os.makedirs(os.path.join(logger.log_dir, "preds"))

if IS_TRAIN:
    trainer = L.Trainer(
        check_val_every_n_epoch = 1,
        logger                  = logger,
        log_every_n_steps       = 1 if IS_VERBOSE else 50,
        max_epochs              = train_hp["max_epochs"],
        callbacks               = [
            EarlyStopping(monitor='val/mse', patience=2, mode='min'),
            ModelCheckpoint(monitor='val/mse', mode="min", save_top_k=1, verbose=True)
        ]
    )
else:
    trainer = L.Trainer(
        check_val_every_n_epoch = 1,
        logger                  = logger,
        log_every_n_steps       = 1 if IS_VERBOSE else 50,
        limit_train_batches     = 50,
        limit_val_batches       = 50,
        limit_test_batches      = 50,
        max_steps               = 50,
        deterministic           = True
    )


# ---------------------------------- Training ------------------------------------
model = LitLRModel(**model_hp)
dm = gray_SingleImageRoundshotDataModule(data_dir="/home/masc12/dev/masc12-mthesis/data/poc", **other_hp)

print(f"Finished setup {(dt.datetime.now() - execution_time).seconds} seconds.\nStarting training...")

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

print(f"Finished training after {((dt.datetime.now() - execution_time)).seconds//60} mins.")


