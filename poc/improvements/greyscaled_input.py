import sys, torch, os, datetime as dt, numpy as np, cv2, pandas as pd
import lightning as L, torch.nn.functional as F, torchvision.transforms as transforms
from torch import nn
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from poc.dataloader import RoundshotDataModule, pvImgSeqDataset
from models.LitBaseModel import LitBaseModel
from models.RsModel import RsModel, RsCNN, RsLSTM

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

# ----------------------------------- Model --------------------------------------
# seed_everything(42)

class PMB_RSModel(RsModel):
    def __init__(self, img_size=(1, 250, 250), patch_size=25, lstm_out_size=64, num_conv_blocks=2, final_actfn=torch.nn.Identity()):
        super().__init__()

        # Adaptive pooling to segment into patches
        self.patch_size = patch_size
        self.avg_pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size[1] // patch_size) # number of patches in one dimension

        # CNN to extract features from patches
        self.cnn = RsCNN(
            img_size          = img_size,
            num_conv_blocks   = num_conv_blocks,
            conv_kernel_size  = 3,
            activation_fn     = F.relu,
        )

        # Compute the output size of the CNN
        heightwidth  = (num_patches - 4 * (2**num_conv_blocks - 1)) // (2**num_conv_blocks)
        channels = 2**(num_conv_blocks + 3)
        
        self.lstm = RsLSTM(
            in_dim  = channels*(heightwidth**2), 
            out_dim = lstm_out_size
        )
        self.fc = nn.Linear(lstm_out_size, 1)
        self.activation_fn = final_actfn

    def forward(self, x):
        # handling image sequences in cnn: https://discuss.pytorch.org/t/how-to-input-image-sequences-to-a-cnn-lstm/89149/2
        bat_len, seq_len = x.shape[0], x.shape[1]
        cnn_in   = x.view(-1, *x.shape[2:])                     # cnn_in.shape = [B*seq,C,H,W]
        
        patch_means = self.avg_pool(cnn_in)
        cnn_out  = self.cnn(patch_means)
        
        lstm_in  = cnn_out.view(bat_len, seq_len, -1)           # lstm_in.shape = [B,seq,C*H*W]
        lstm_out = self.lstm(lstm_in)
        
        fc_in    = lstm_out[:, -1, :]                           # fc_in.shape = [B,out_dim]
        out      = self.fc(fc_in)                               # ↪ use only the last time step's output 
        return self.activation_fn(out)                          # ↪ (i.e. the final contextualized representation of the sequence)

class Lit_PMB_RsModel(LitBaseModel):
    def initialize_model(self, **kwargs):
        return PMB_RSModel(**kwargs)

class gray_pvImgSeqDataset(pvImgSeqDataset):
    def __getitem__(self, seq_idx):
        """ Get the image sequence and the target value for the given index. """
        images = []
        start_idx = self.lookup_table[seq_idx]
        end_idx = start_idx + self.seq_length 
        
        # target value
        y_idx = (end_idx-1) + self.lead_time  # idx of target value after 'lead_time' steps
        ts = int(self.annotations.iloc[y_idx]["to"].timestamp()) # timestamp of the target value
        y_scaled = (self.annotations.iloc[y_idx]["energy-produced-Wh"]-self.ymin) / (self.ymax-self.ymin)
        y_scaled = torch.tensor(float(y_scaled))

        # model input 
        for i in range(start_idx, end_idx): # end_idx is exclusive
            if self.annotations.iloc[i]["has_image"]: # has image
                img_path = os.path.join(self.img_dir, self.annotations.iloc[i]["to"].strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
                img = cv2.imread(img_path)
            
            elif self.annotations.iloc[i]["atday"]: # no image + is at day
                img = np.ones((250, 250, 3), dtype = np.uint8) * 255 # Create a white image
            
            else: # no image + is night
                img = np.zeros((250, 250, 3), dtype = np.uint8)  # Create a black image
            
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

            if self.transform:
                img = self.transform(img)
            
            images.append(img)
        image_sequence = torch.stack(images)
        return (image_sequence, y_scaled, ts)

class gray_RoundshotDataModule(RoundshotDataModule):
    def setup(self, stage: str):
        """ Setup the data split for the given stage ('fit' or 'test'). """
        if stage == "fit":
            self.train_dataset = gray_pvImgSeqDataset(os.path.join(self.data_dir, "train_y.csv"), self.img_dir, self.transform, self.seq_length, self.lead_time, mode="train")
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.val_dataset = gray_pvImgSeqDataset(os.path.join(self.data_dir, "val_y.csv"), self.img_dir, self.transform, self.seq_length, self.lead_time, ymin, ymax, mode="val")

        elif stage == "test":
            ymin, ymax = self.train_dataset.ymin, self.train_dataset.ymax
            self.test_dataset = gray_pvImgSeqDataset(os.path.join(self.data_dir, "test_y.csv"), self.img_dir, self.transform, self.seq_length, self.lead_time, ymin, ymax, mode="test")

# ------------------------------ Hyperparameters ---------------------------------
model_hp = {
    "lr": 0.001,
    "optimizer": torch.optim.Adagrad,
    "verbose": IS_VERBOSE,
    "loss_outlier_threshold": 4e5 if IS_VERBOSE else sys.float_info.max,
    "img_size": (1, 250, 250), 
    "patch_size": 5, 
    "num_conv_blocks": 3, 
    "final_actfn": nn.ReLU()
}

other_hp = {
    "batch_size":       8,                         
    "seq_length":       6,                          # number of images in a sequence 
    "lead_time": 1,                          # number of steps (10 minute) ahead we want to predict
    "num_workers":      15 if IS_TRAIN else 0,     
    "transform":        transforms.ToTensor(),  	# converts a image/np.ndarray (HxWxC) in the range [0, 255] to a torch.FloatTensor of shape (CxHxW) w/ range [0.0, 1.0]
    "shuffle_train":    True,
}

train_hp = {
    "max_epochs": 15, 
}


# -------------------------------- Setup logging ---------------------------------
print("RUN NAME: ", RUN_NAME)
logger = TensorBoardLogger(save_dir="poc/tblogs/other", name="gray_patched_LitRsModel", version=RUN_NAME, default_hp_metric=False)
hparams = {
    k: str(v) if not isinstance(v, (int, float, bool, tuple, list, dict, set)) else v
    for k, v in {**model_hp, **other_hp, **train_hp}.items()
}
logger.log_hyperparams(hparams, {"hp/val_rmse": 0, "hp/val_mae": 0})
predpath = os.path.join(logger.log_dir, "preds")
os.makedirs(predpath)

if IS_TRAIN:
    trainer = L.Trainer(
        check_val_every_n_epoch = 1,
        logger                  = logger,
        log_every_n_steps       = 1 if IS_VERBOSE else 50,
        max_epochs              = train_hp["max_epochs"],
        callbacks               = EarlyStopping(monitor='val/mse', patience=2, mode='min')
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
model = Lit_PMB_RsModel(**model_hp)
dm = gray_RoundshotDataModule(data_dir="/home/masc12/dev/masc12-mthesis/data/poc", **other_hp)

# dm.prepare_data(resizeAll=False, split=True)

print(f"Finished setup {(dt.datetime.now() - execution_time).seconds} seconds.\nStarting training...")

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

print(f"Finished training after {((dt.datetime.now() - execution_time)).seconds//60} mins.")


# --------------------------------- Save model -----------------------------------
if IS_TRAIN:
    model_path = os.path.join(logger.log_dir, "model")
    os.makedirs(model_path)
    torch.save(model.model.state_dict(), os.path.join(model_path, "model.pt"))
