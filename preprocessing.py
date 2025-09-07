import torch
import os
import numpy as np
import pandas as pd
df = pd.read_csv('data1.csv')

import numpy as np

#data preprocessing
import ast
wave_columns = ['delta','highAlpha', 'highBeta',
'lowAlpha', 'lowBeta', 'lowGama', 'theta']

df = df.drop(index=77).reset_index(drop=True)
def parse_list_of_floats(cell):
    try:
        values = ast.literal_eval(cell)
        return [float(x) for x in values]
    except Exception:
        return [np.nan] * 179  # ensures consistent shape

#Apply parsing and expand each EEG feature
for col in wave_columns:
    df[col] = df[col].apply(parse_list_of_floats)


eeg_tensor = np.stack([np.stack(df[col].values) for col in wave_columns], axis=1).astype(np.float32)
print("Before normalization:", eeg_tensor.shape)

#Normalize per-sample across each channel (along axis=2)

for i, col in enumerate(wave_columns):
    df[col] = list(eeg_tensor[:, i, :])

def log_z_normalize_column(col_data):
    """Apply log1p followed by z-score to each 180D vector."""
    return col_data.apply(lambda x: (
        (np.log1p(np.array(x, dtype=np.float32)) -
         np.log1p(np.array(x, dtype=np.float32)).mean()) /
         np.log1p(np.array(x, dtype=np.float32)).std()
    ))
#Apply to each wave column
for col in wave_columns:
    df[col] = log_z_normalize_column(df[col])
    print(df[col].dtype)

#eeg_min = eeg_tensor.min(axis=2, keepdims=True)
#eeg_max = eeg_tensor.max(axis=2, keepdims=True)
#eeg_tensor = (eeg_tensor - eeg_min) / (eeg_max - eeg_min + 1e-8)

#for i, col in enumerate(wave_columns):
    #df[col] = list(eeg_tensor[:, i, :])

# Extract labels (actual age)
df['user_birth_data'] = pd.to_datetime(df['user_birth_data'])

# Get today's date
today = pd.to_datetime('today')

# Compute age in years
df['actual_age'] = (today - df['user_birth_data']).dt.days // 365
y_age = df['actual_age'].values.astype(np.float32)

y_mean = df['actual_age'].mean()
y_std = df['actual_age'].std()


#Final check
print("After normalization:", eeg_tensor.shape)
print("Example tensor (first sample, first channel):", eeg_tensor[0, 0, :5])
print("Target shape:", y_age.shape)
df["highAlpha"].iloc[0].dtype

#create dataset
X = torch.tensor(df[wave_columns].values.tolist()) #shape is [number of patients, 7, 180]
# Extract labels (actual age)
df['user_birth_data'] = pd.to_datetime(df['user_birth_data'])
torch.tensor(X)

# Get today's date
today = pd.to_datetime('today')

# Compute age in years
df['actual_age'] = (today - df['user_birth_data']).dt.days // 365

#normalize ages

y_mean = df['actual_age'].mean()
y_std = df['actual_age'].std()


df['age_normalized'] = (df['actual_age'] - y_mean) / y_std


Y = df['age_normalized'].values.astype(np.float32)
