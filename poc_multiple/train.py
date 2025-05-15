import sys, torch, os, datetime as dt, pandas as pd, time
import lightning as L, torchvision.transforms.v2 as v2
from math import floor
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import itertools, inspect, types, re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poc_multiple.multiple_dataloader import RoundshotMultipleDataModule
from models.LitBaseModel import LitBaseModel
from models.RsMultipleLateFusionAttnModel import RsMultipleLateFusionAttnModel

torch.set_float32_matmul_precision("high")

webcams = ['altenrhein', 'ebenalp', 'golfclub-erlen', 'kronberg', 'romanshorn', 'rorschacherberg', 'sitterdorf', 'wiler-turm']
for j in range(1,2):
    webcam_pairs = itertools.combinations(webcams, 2)
    for wbc in list(webcam_pairs) + webcams:
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

        # ------------------------------ Hyperparameters ---------------------------------

        model_hp = {
            "lr":                       0.001,
            "conv_init":               "kaiming_uniform",    # default: "kaiming_uniform"
            "lstm_init":               "zeros",              # default: "zeros"
            "fc_init":                 "xavier_uniform",     # default: "xavier_uniform"
            "optimizer":                torch.optim.Adagrad,
            "verbose":                  IS_VERBOSE,
            "loss_outlier_threshold":   2 if IS_VERBOSE else sys.float_info.max,
            "locations":                list(wbc) if type(wbc) == tuple else [wbc]
        }


        # IS_VERBOSE = True if j%2 == 0 else False

        # to fix the #img per batch, we need to set the batch size to a multiple of the number of locations
        # this holds, given that the sequence length is the same for all locations
        # seq_length * batch_size * num_locations = #img_per_batch
        # img_per_batch can not exceed 90 per GPU limitations
        other_hp = {
            "batch_size":       floor(18 / len(model_hp["locations"])),  # batch size for training
            "seq_length":        5,                         # number of images in a sequence 
            "lead_time":         1, # × 10 = [min]          # number of time steps we want to predict ahead (1 step = 10 min)
            "num_workers":      15 if IS_TRAIN else 0,      # number of workers for dataloader
            "shuffle_train":    True,                       # shuffle the training data (shuffles sequences; not images within a sequence)
            "transform":        v2.Compose([
                                    v2.ToImage(),               # Converts input to a tensor image of shape (CxHxW)
                                    v2.ToDtype(torch.float32, scale=True),  # Converts to float and scales to [0, 1]
                                ]),
            "logtransform":     False,                      # apply log transformation to the target variable
            "pvhistory":        False,
            "sunpos":           False
        }

        train_hp = {
            "max_epochs": -1, 
            "seed": 41,
        }

        # --------------------------------- Skip if already trained ---------------------------------
        smloc = "singleloc" if len(model_hp["locations"]) == 1 else "multiloc"
        try:
            exp = os.listdir(f"poc_multiple/tblogs/fh{j}0-LitRsMultipleModel/RsMultipleLateFusionAttnModel2/{smloc}")
            exp = [e.split("_") for e in exp]
            exp = [tuple(e[:2]) if len(e) == 4 else e[0] for e in exp]
            if wbc in exp: continue
        except FileNotFoundError:
            pass


        # -------------------------------- Setup logging ---------------------------------
        RUN_NAME = RUN_NAME.split("_")
        RNLOCS = "_".join(model_hp["locations"])
        if len(RUN_NAME) == 3: # debug
            RUN_NAME = f"{RUN_NAME[0]}_{RNLOCS}_{RUN_NAME[1]}_{RUN_NAME[2]}"
        elif len(RUN_NAME) == 2: # train
            RUN_NAME = f"{RNLOCS}_{RUN_NAME[0]}_{RUN_NAME[1]}"

        print("\nRUN NAME: ", RUN_NAME)
        print(f"fh: {j}0")

        logger = TensorBoardLogger(save_dir="poc_multiple/tblogs", name=f"fh{other_hp['lead_time']}0-LitRsMultipleModel/RsMultipleLateFusionAttnModel2/{smloc}", version=RUN_NAME, default_hp_metric=False)

        def serialize_value(v):
            if isinstance(v, (int, float, bool, tuple, list, dict, set, str)):
                return v
            elif isinstance(v, types.LambdaType) and v.__name__ == "<lambda>":
                try:
                    src = inspect.getsource(v).strip()
                    match = re.search(r'lambda.*', src)
                    return match.group(0).split("#")[0].strip() if match else "<lambda source unavailable>"
                except (OSError, TypeError):
                    return "<lambda source unavailable>"
            else:
                return str(v)

        hparams = {
            k: serialize_value(v)
            for k, v in {**model_hp, **other_hp, **train_hp}.items()
        }

        logger.log_hyperparams(hparams, {"hp/val_total_rmse": 0, "hp/val_total_mae": 0})
        predpath = os.path.join(logger.log_dir, "preds")
        os.makedirs(predpath)

        if IS_TRAIN:
            trainer = L.Trainer(
                check_val_every_n_epoch = 1,
                logger                  = logger,
                max_epochs              = train_hp["max_epochs"],
                log_every_n_steps       = 1 if IS_VERBOSE else 50,
                # gradient_clip_val       = 1,
                callbacks               = [
                    EarlyStopping(monitor='val/total_mse', patience=6, mode='min'),
                    #ModelCheckpoint(monitor='val/total_mse', mode='min', save_top_k=1, verbose=True)
                ],
                enable_checkpointing    = False
            )
        else:
            trainer = L.Trainer(
                check_val_every_n_epoch = 1,
                logger                  = logger,
                max_epochs              = train_hp["max_epochs"],
                log_every_n_steps       = 1 if IS_VERBOSE else 50,
                limit_train_batches     = 50,
                limit_val_batches       = 50,
                limit_test_batches      = 50,
                max_steps               = 50,
                deterministic           = True,
                enable_checkpointing    = False
            )

        # ----------------------------------- Model --------------------------------------
        seed_everything(train_hp["seed"])

        class LitRsModel(LitBaseModel):
            def initialize_model(self, **kwargs):
                return RsMultipleLateFusionAttnModel(**kwargs)

        # ---------------------------------- Training ------------------------------------
        print(f"Model will be trained on {len(model_hp['locations'])} locations: {model_hp['locations']}.")
        model = LitRsModel(**model_hp)
        dm = RoundshotMultipleDataModule(data_dir="data/poc_multiple", locations=model_hp['locations'], **other_hp)
        dm.prepare_data(resizeAll=False, split=True)
        print(f"Finished setup in {(dt.datetime.now() - execution_time).seconds} seconds.")

        print(f"Starting training on:", torch.cuda.get_device_name(0))
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
        print(f"Finished training after {((dt.datetime.now() - execution_time)).seconds//60} mins.")

        # >>> mthesis_env/bin/python poc_multiple/train.py train verbose
