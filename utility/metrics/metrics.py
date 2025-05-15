import pandas as pd, numpy as np

def compute_metrics(df:pd.DataFrame, production_max:float, prefix:str="") -> dict:
    """
    Compute metrics for the model.

    Args:
    df: pd.DataFrame: DataFrame containing the predictions and the ground truth.
    - df["timestamp"]: Timestamp of the prediction.
    - df["y_true"]: Ground truth.
    - df["y_pred"]: Predictions.
    - df["batch"]: Batch number.
    production_max: Maximum production value a panel can produce. Unit: Wh.

    Returns:
    dict: Dictionary containing the computed metrics.
    - "mse": Mean Squared Error.
    - "mae": Mean Absolute Error.
    - "rmse": Root Mean Squared Error.
    - "nmae%": Normalized Mean Absolute Error [in %]. 
    - "nrmse%": Normalized Root Mean Squared Error [in %].
    - "mape%": Mean Absolute Percentage Error [in %].
    - "wape%": Weighted Absolute Percentage Error [in %].
    """

    mae   =  (df["y_pred"] - df["y_true"]).abs().mean()
    mse   = ((df["y_true"] - df["y_pred"])**2).mean()
    rmse  = np.sqrt(mse)
    nrmse =  (rmse / production_max) * 100
    nmae  =  (mae / production_max) * 100
    
    eps   = 1e-16
    # mape  = ((df["y_pred"] - df["y_true"]) / (df["y_true"]+eps)).abs().mean() * 100
    wape  =  (df["y_pred"] - df["y_true"]).abs().sum() / (df["y_true"]+eps).abs().sum() * 100 

    return {
        f"{prefix}mae": mae,
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}nmae%": nmae,
        f"{prefix}nrmse%": nrmse,
       #f"{prefix}mape%": mape,
        f"{prefix}wape%": wape
    }
