import sys, torch, time, os, datetime as dt
import lightning as L, torchvision.transforms as transforms
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poc.dataloader import RoundshotDataModule
from models.LitBaseModel import LitBaseModel
from models.RsModel import RsModel

# from torchinfo import summary
# summary(RsModel(), input_size=(6, 8, 3, 250, 250))

torch.set_float32_matmul_precision("high")


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
print("\n\nRUN NAME: ", RUN_NAME)


# ------------------------------ Hyperparameters ---------------------------------
model_hp = {
    "lr":                       0.001,
    "conv_init":               "kaiming_uniform",    # default: "kaiming_uniform"
    "lstm_init":               "zeros",              # default: "zeros"
    "fc_init":                 "xavier_uniform",     # default: "xavier_uniform"
    "optimizer":                torch.optim.Adagrad,
    "verbose":                  IS_VERBOSE,
    "loss_outlier_threshold":   4e5 if IS_VERBOSE else sys.float_info.max,
}

other_hp = {
    "batch_size":        8,                         # number of sequences in a batch
    "seq_length":        5,                         # number of images in a sequence 
    "lead_time":  1, # horizon × 10 = [min]  # number of time steps we want to predict ahead (1 step = 10 min)
    "num_workers":      15 if IS_TRAIN else 0,      # number of workers for dataloader
    "transform":        transforms.ToTensor(),  	# converts a image/np.ndarray (HxWxC) in the range [0, 255] to a torch.FloatTensor of shape (CxHxW) w/ range [0.0, 1.0]
    "shuffle_train":    True,                       # shuffle the training data (shuffles sequences; not images within a sequence)
}

train_hp = {
    "max_epochs": 15, 
    "seed": 40,
}


# -------------------------------- Setup logging ---------------------------------
logger = TensorBoardLogger(save_dir="poc/tblogs", name="LitRsModel", version=RUN_NAME, default_hp_metric=False)
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

# ----------------------------------- Model --------------------------------------
seed_everything(train_hp["seed"])

class LitRsModel(LitBaseModel):
    def initialize_model(self, **kwargs):
        return RsModel(**kwargs)

# ---------------------------------- Training ------------------------------------
model = LitRsModel(**model_hp)
dm = RoundshotDataModule(data_dir="/home/masc12/dev/masc12-mthesis/data/poc", **other_hp)

dm.prepare_data(resizeAll=False, split=True)

print(f"Finished setup in {(dt.datetime.now() - execution_time).seconds} seconds.")
print(f"Starting training...")

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

print(f"Finished training after {((dt.datetime.now() - execution_time)).seconds//60} mins.")