import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# find optimal learning rate
from lightning.pytorch.tuner import Tuner

# Load data
data_path = "/home/yasser/SM_project/src/data/time_series_data.csv"
data = pd.read_csv(data_path)

# Reset index to make "DATE" a column if it's not already one
data.rename(columns={'Unnamed: 0': 'DATE'}, inplace=True)

# Ensure 'DATE' is datetime and sort
data['DATE'] = pd.to_datetime(data['DATE'])
data.sort_values('DATE', inplace=True)


# Exclude target variable and non-numeric columns for scaling
features_to_scale = data.columns.difference(['NEXT_DAY_ADJUSTED_CLOSING_PRICE', 'DATE', 'SYMBOL'])

# Scaling features
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Create 'time_idx' column with continuous indexing
data['time_idx'] = data['DATE'].rank(method='dense').astype(int) - data['DATE'].rank(method='dense').astype(int).min()

print(data)

# Setup TimeSeriesDataSet
training_cutoff = data['time_idx'].max() - 1

training = TimeSeriesDataSet(
    data=data[data.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="NEXT_DAY_ADJUSTED_CLOSING_PRICE",
    group_ids=["SYMBOL"],
    min_encoder_length=12,  # Assuming at least 1 year of history
    max_encoder_length=36,  # Assuming up to 3 years of history
    min_prediction_length=1,
    max_prediction_length=1,  # Predicting the next day
    static_categoricals=["SYMBOL"],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=list(features_to_scale),
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
    target_normalizer=GroupNormalizer(groups=["SYMBOL"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation set and dataloaders
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

batch_size = 128  # Adjust based on your system's capability
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
# baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
# MAE()(baseline_predictions.output, baseline_predictions.y)


# configure network and trainer
pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator="cpu",
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=8,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    loss=QuantileLoss(),
    optimizer="Ranger"
    # reduce learning rate if no improvement in validation loss after x epochs
    # reduce_on_plateau_patience=1000,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")



res = Tuner(trainer).lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)