import pandas as pd
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

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
