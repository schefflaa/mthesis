import os, sys
import pandas as pd
import numpy as np
import pvlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.metrics.metrics import compute_metrics

# --- CONFIGURATION ---
latitude = 47.424492554512014
longitude = 9.376722938498643
timezone = "Europe/Zurich"

# --- LOAD DATA ---
train_df = pd.read_csv('data/poc_multiple/train_y.csv', sep=";", encoding="utf-8", parse_dates=["to"])
val_df = pd.read_csv('data/poc_multiple/val_y.csv', sep=";", encoding="utf-8", parse_dates=["to"])

# Concatenate to ensure continuity if necessary
full_df = pd.concat([train_df, val_df]).reset_index(drop=True)

# Ensure 'to' column is a DatetimeIndex
full_df['to_(naive)'] = pd.to_datetime(full_df['to_(naive)'], utc=True)
full_df.set_index('to_(naive)', inplace=True)  # Set 'to' as the index

# --- CLEAR-SKY MODEL ---
location = pvlib.location.Location(latitude, longitude, tz=timezone)
clearsky = location.get_clearsky(full_df.index)  # Use DatetimeIndex

# Reset index after clearsky calculation if needed
full_df.reset_index(inplace=True)

# Convert observed Wh/10min to kW
full_df['power'] = full_df['energy-produced-Wh'] * 6 / 1000  # (Wh → kW)

# Normalize clear-sky power to max observed power
max_power = full_df['power'].max()
full_df['clear_sky_power'] = clearsky['ghi'].values * (max_power / clearsky['ghi'].max())

# --- SMART PERSISTENCE MODEL (safe version) ---
shifted_power = full_df['power'].shift(1)
shifted_clear_sky = full_df['clear_sky_power'].shift(1)

# Avoid division by zero or NaN
valid_mask = shifted_clear_sky > 0
full_df['smart_persistence'] = np.where(
    valid_mask,
    shifted_power * (full_df['clear_sky_power'] / shifted_clear_sky),
    np.nan
)

# Split back into train/validation (aligned correctly)
val_start = train_df.shape[0] + 1
val_pred = full_df.loc[val_start:, 'smart_persistence'].reset_index(drop=True)
val_true = full_df.loc[val_start:, 'power'].reset_index(drop=True)

# --- EVALUATE ---
metrics = compute_metrics(
    pd.DataFrame({
        'y_true': val_true,
        'y_pred': val_pred
    }),
    production_max=max_power,
    prefix='smart_persistence_'
)

print(f'Smart Persistence metrics:')
for metric, value in metrics.items():
    print(f'{metric}: {value:.4f}')