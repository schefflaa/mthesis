import pandas as pd
import numpy as np
import os, yaml, json
import datetime as dt

def get_results(folder, fh=None):
    experiments = sorted(os.listdir(folder))
    experiments = [exp for exp in experiments if "." not in exp]  # only keep folders
    rmses = [np.inf] * len(experiments)
    metrics = [None] * len(experiments)
    best_epochs = [np.inf] * len(experiments)

    ts = []
    location_amounts = []
    locations = []

    for i, exp in enumerate(experiments):
        # load locations
        yaml_file = os.path.join(folder, exp, 'hparams.yaml')
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        loc = config.get('locations')
        location_amounts.append(len(loc))
        locations.append(loc)

        fn = "_".join(exp.split("_")[-2:])
        ts.append(dt.datetime.strptime(fn, "%Y-%m-%d_%H-%M-%S"))

        # load rmse + mae
        if fh is None:
            predfolder = os.path.join(folder, exp, 'preds')
        else:
            predfolder = os.path.join(folder, exp, 'preds', f"fh{fh}")
        
        predfiles = os.listdir(predfolder)
        predfiles = [f for f in predfiles if f.startswith('val') and f.endswith('.csv')]
        epochs = [f.split("_")[0][3:] for f in predfiles]

        # get best epoch
        for e in epochs:
            metricsfile = os.path.join(predfolder, f'val{e}_stats.json')
            with open(metricsfile, 'r') as file:
                res = json.load(file)

                if rmses[i] > res['total_rmse']:
                    rmses[i] = res['total_rmse']
                    metrics[i] = res
                    best_epochs[i] = e

    return pd.DataFrame({
        "timestamp":            ts,
        "location_amount":      location_amounts,
        "locations":            locations,
        "best_epoch":           best_epochs,
        "best_epoch_metrics":   metrics
    })
