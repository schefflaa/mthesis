import os, sys, json, torch
import torch.nn.functional as F, lightning as L, pandas as pd, numpy as np, matplotlib.pyplot as plt
from utility.metrics.statistical_metrics import compute_statistics
from utility.metrics.metrics import compute_metrics


class plainLSTMbaseModel(L.LightningModule):
    def __init__(self, lr=0.001, optimizer=torch.optim.Adagrad, verbose=False, loss_outlier_threshold=sys.float_info.max, **kwargs):
        super().__init__()
        self.model = self.initialize_model(**kwargs)
        self.loss_outlier_threshold = loss_outlier_threshold
        self.optimizer = optimizer
        self.verbose = verbose
        self.lr = lr

    def initialize_model(self, **kwargs): # Abstract method. Implemented by the subclass
        raise NotImplementedError("method 'initialize_model' not implemented")

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """ Perform a single training step. """
        _, pv_history, y, y_ts, timesteps = batch
        out = self.model(pv_history)
        y_pred = out.squeeze(-1) # remove dimension [batch, 1] -> [batch]
        
        # Rescale the predictions and target values
        min_val, max_val = self.trainer.datamodule.min_value, self.trainer.datamodule.max_value
        y_true_rescaled = y * (max_val - min_val) + min_val
        y_pred_rescaled = y_pred * (max_val - min_val) + min_val

        # Compute loss and log it
        loss = F.mse_loss(y_pred_rescaled, y_true_rescaled)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch,  batch_idx):
        return self.batch_eval(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.batch_eval(batch, batch_idx, "test")
    
    def batch_eval(self, batch, batch_idx, mode=None):
        """ 
        Evaluates the model on the given batch. 
        Saves the batch predictions to file for later aggregation.
        Gets called during validation and test steps.
        """
        # Forward pass
        _, pv_history, y, y_ts, timesteps = batch
        out = self.model(pv_history)
        y_pred = out.squeeze(-1) # remove dimension [batch, 1] -> [batch]
        
        # Rescale the predictions and the target values
        min_val, max_val = self.trainer.datamodule.min_value, self.trainer.datamodule.max_value
        y_true_rescaled = y * (max_val - min_val) + min_val
        y_pred_rescaled = y_pred * (max_val - min_val) + min_val
        y_pred_clamped = torch.clamp(y_pred_rescaled, min=0) # Ensure the target values are non-negative
        
        # Save the predictions to file for plotting
        os.makedirs(os.path.join(self.logger.log_dir, "preds"), exist_ok=True) # Create the directory if it does not exist, if it does, do nothing
        predictions_path = os.path.join(self.logger.log_dir, "preds" , f'{mode}_pred_b{batch_idx}.csv')
        pd.DataFrame({
            "timestamp": y_ts.detach().cpu().numpy(),
            "y_true": y_true_rescaled.detach().cpu().numpy(),
            "y_pred": y_pred_clamped.detach().cpu().numpy(),
            "batch": batch_idx
        }).to_csv(predictions_path, sep=";", index=False, encoding="utf-8")

    def on_validation_epoch_end(self):
        """
        Gets called at the end of each validation epoch.
        Evaluates the model on the validation set and logs the results.
        """
        if not self.trainer.sanity_checking:
            metrics = self.full_eval(mode="val", epoch=self.trainer.current_epoch)
            for name, value in metrics.items():
                self.log(f'val/{name}', value, prog_bar=(name=="total_rmse"))
                self.log(f'hp/val_{name}', value)

    def on_test_end(self):
        """
        Gets called at the end of all test_step.
        Evaluates the model on the test set and logs the results.
        """
        metrics = self.full_eval(mode="test")
        for name, value in metrics.items():
                self.logger.experiment.add_scalar(f'test/{name}', value, global_step=self.global_step)


    def full_eval(self, mode:str, epoch:str="") -> dict:
        """ 
        Gets called at the end of each validation and test epoch.
        Aggregates the prediction batches and plots them. 
        Calculates various metrics, logs them to TensorBoard and saves them to file.
        `mode` is either 'val' or 'test'.
        """
        predictions_path = os.path.join(self.logger.log_dir, "preds")
        preds_df = aggregate_predictions(predictions_path, mode, epoch)
        
        if mode == "test":
            metrics = compute_metrics(preds_df, production_max=21_780*(1/6))
            plot_predictions(preds_df, metrics, predictions_path, mode, epoch)
        elif mode == "val":
            preds_list = split_predictions(preds_df)
            metrics = {}
            for i, part in enumerate(preds_list):
                metrics.update(compute_metrics(part, production_max=21_780*(1/6), prefix=f"{i}_"))
            metrics.update(compute_metrics(preds_df, production_max=21_780*(1/6), prefix="total_"))

            plot_predictions_with_subplots(preds_list, metrics, predictions_path, mode, epoch)

        statistics = compute_statistics(preds_df)
        statistics_path = os.path.join(predictions_path, f"{mode}{epoch}_stats.json")
        with open(statistics_path, "w", encoding="utf-8") as f:
            json.dump(metrics|statistics, f, indent=4, ensure_ascii=False)
        return metrics

            


# ------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------                 Helper Functions                ------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------- #


def aggregate_predictions(predpath:str, mode:str, epoch:str="") -> pd.DataFrame:
    """
    Concatenate the individual batch prediction files into a single DataFrame.
    Save the concatenated DataFrame to a single file.
    Delete the individual batch prediction files.
    """
    all_files = os.listdir(predpath)
    all_files = [os.path.join(predpath, x) for x in all_files if x.startswith(f"{mode}_pred_b")]
    concatenated_df = pd.concat([pd.read_csv(file, sep=";", encoding="utf-8") for file in all_files])
    concatenated_df["timestamp"] = concatenated_df["timestamp"].apply(lambda x: pd.to_datetime(x, unit='s', utc=True).tz_convert("Europe/Zurich"))
    concatenated_df.sort_values("timestamp", inplace=True)
    concatenated_df.reset_index(drop=True, inplace=True)
    
    final_pred_file = os.path.join(predpath, f"{mode}{epoch}_predictions.csv")
    
    concatenated_df.to_csv(final_pred_file, sep=";", index=False, encoding="utf-8")

    for file in all_files:
        os.remove(file)

    return concatenated_df

def split_predictions(preds:pd.DataFrame) -> list:
    # Compute time differences
    time_diffs = [preds["timestamp"][1:].iloc[i] - preds["timestamp"][:-1].iloc[i] for i in range(len(preds["timestamp"][:-1]))]
    time_diffs = pd.Series(time_diffs)	

    # find the two largest time gaps
    largest_gaps = time_diffs.nlargest(2).index

    # Sort indices to split in order
    split_idx1, split_idx2 = sorted(largest_gaps)

    # Split the series
    part1 = preds[:split_idx1 + 1]
    part2 = preds[split_idx1 + 1:split_idx2 + 1]
    part3 = preds[split_idx2 + 1:]
    return [part1, part2, part3]

def plot_predictions(predictions:pd.DataFrame, metrics:dict, predpath:str, mode:str, epoch:str="") -> None:
    """ Plot the predictions against the true values. """
    plt.figure(figsize=(18, 6))

    for idx, row in predictions.iterrows():
        color = "grey" #if row["y_pred"] < row["y_true"] else "red"
        plt.plot([row["timestamp"], row["timestamp"]], [row["y_true"], row["y_pred"]], color=color, linewidth=.8, alpha=0.5)

    plt.plot(predictions["timestamp"], predictions["y_true"])
    plt.plot(predictions["timestamp"], predictions["y_pred"])

    plt.scatter(predictions["timestamp"], predictions["y_true"], s=5, label="True")
    plt.scatter(predictions["timestamp"], predictions["y_pred"], s=5, label="Predicted")

    plt.title(f"Model Predictions on {mode} data set. RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}", fontsize=12, weight="bold", y=1)
    plt.ylabel('Energy [Wh]')
    plt.xlabel('Time')
    plt.legend(markerscale=2)
    plt.grid()
    plt.savefig(os.path.join(predpath, f"{mode}{epoch}_predictions.png"))
    plt.close()

def plot_predictions_with_subplots(predictions:list[pd.DataFrame], metrics:dict, predpath:str, mode:str, epoch:str="") -> None:
    """ Plot the predictions against the true values. """
    n_subplot = len(predictions)
    maxval = max(part[["y_true", "y_pred"]].max().max() for part in predictions)

    fig, axs = plt.subplots(n_subplot, 1, figsize=(18, 2 + 6*n_subplot))
    if n_subplot == 1:
        axs = [axs]
    for i, (pred, ax) in enumerate(zip(predictions, axs)):
        # vertical errorbar between pred and true
        for idx, row in pred.iterrows():
            color = "grey" #if row["y_pred"] < row["y_true"] else "red"
            ax.plot([row["timestamp"], row["timestamp"]], [row["y_true"], row["y_pred"]], color=color, linewidth=.8, alpha=0.5)
        
        ax.plot(pred["timestamp"], pred["y_true"])
        ax.plot(pred["timestamp"], pred["y_pred"])

        ax.scatter(pred["timestamp"], pred["y_true"], label="True", s=15)
        ax.scatter(pred["timestamp"], pred["y_pred"], label="Predicted", s=15)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy [Wh]")
        ax.set_ylim(0, maxval+100)
        
        ax.set_title(f"RMSE: {metrics[f'{i}_rmse']:.2f}, MAE: {metrics[f'{i}_mae']:.2f}")
        ax.legend(markerscale=2)
        ax.grid()
    
    plt.suptitle(f"Model Predictions on {mode} epoch {epoch}    RMSE: {metrics['total_rmse']:.2f}, MAE: {metrics['total_mae']:.2f}", fontsize=16, weight="bold", y=.99)
    plt.tight_layout()
    plt.savefig(os.path.join(predpath, f"{mode}{epoch}_predictions.png"))
    plt.close()