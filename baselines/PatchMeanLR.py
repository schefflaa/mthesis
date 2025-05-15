import sys, torch, os, cv2, datetime as dt, numpy as np
import lightning as L, torch.nn as nn, torchvision.transforms as transforms
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import SingleImageRoundshotDataModule
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


# ------------------------------ Model definition --------------------------------
class PatchMeanLRModel(nn.Module):
    def __init__(self, input_size, patch_size):
        """
        input_size: size of the input image (assuming square image)
        patch_size: size of the patches to segment the image into
        """

        super(PatchMeanLRModel, self).__init__()
        self.patch_size = patch_size
        
        # Adaptive pooling to segment into patches
        self.avg_pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        
        # Fully connected layer for output
        num_patches = (input_size // patch_size) ** 2 *3 # Total number of patches
        self.linear = nn.Linear(num_patches, 1)

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size = x.size(0)
        
        # Apply average pooling to compute mean of patches
        patch_means = self.avg_pool(x)  # (batch_size, channels, pooled_h, pooled_w)
        
        if not IS_TRAIN:
            img = x[0].detach().cpu().numpy()
            img = img.clip(0, 1)  # Ensure values are within [0, 1]
            img = (img * 255).astype(np.uint8)  # Use numpy's astype for uint8 conversion
            img = img.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            cv2.imwrite(f"viz/{self.patch_size}original.png", img)
            
            m = patch_means[0].detach().cpu().numpy()
            m = m.clip(0, 1)  # Ensure values are within [0, 1]
            m = (m * 255).astype(np.uint8)  # Use numpy's astype for uint8 conversion
            m = m.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            cv2.imwrite(f"viz/{self.patch_size}patch_means.png", m)
        
        # Flatten the patch means for input to the linear layer
        patch_means_flat = patch_means.view(batch_size, -1)  # (batch_size, num_patches)
        
        # Pass through the linear layer
        output = self.linear(patch_means_flat)
        return output

class LitPMLRModel(LitBaseModel):
    def initialize_model(self, **kwargs):
        return PatchMeanLRModel(**kwargs)


# ------------------------------ Hyperparameters ---------------------------------
model_hp = {
    "lr": 0.001,
    "optimizer": torch.optim.Adagrad,
    "verbose": IS_VERBOSE,
    "loss_outlier_threshold": 4e5 if IS_VERBOSE else sys.float_info.max,
    "input_size": 250, 
    "patch_size": 25
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
logger = TensorBoardLogger(save_dir="baselines/tblogs", name="PatchMeanLR", version=RUN_NAME, default_hp_metric=False)
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
model = LitPMLRModel(**model_hp)
dm = SingleImageRoundshotDataModule(data_dir="/home/masc12/dev/masc12-mthesis/data/poc", **other_hp)

print(f"Finished setup {(dt.datetime.now() - execution_time).seconds} seconds.\nStarting training...")

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

print(f"Finished training after {((dt.datetime.now() - execution_time)).seconds//60} mins.")

