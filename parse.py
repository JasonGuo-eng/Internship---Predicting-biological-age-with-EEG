from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, normaltest

# Example structure: each cell contains a 1D numpy array of length 180

wave_columns = ['delta','highAlpha', 'highBeta',
'lowAlpha', 'lowBeta', 'lowGama', 'theta']

for wave in wave_columns:
    # Stack all 180D vectors into one long 1D array
    wave_data = df[wave].apply(np.array).to_list()  # turns each row into a numpy array
    wave_data = np.concatenate(wave_data)  # flatten to shape (n_patients * 180,)

    # Normality tests
    stat_shapiro, p_shapiro = shapiro(wave_data)  # Shapiro limit is ~5000 samples
    stat_dagostino, p_dagostino = normaltest(wave_data)

    print(f"\nWave: {wave}")
    print(f"Shapiro-Wilk p = {p_shapiro:.4f} → {'Normal' if p_shapiro > 0.05 else 'Not Normal'}")
    print(f"D’Agostino p = {p_dagostino:.4f} → {'Normal' if p_dagostino > 0.05 else 'Not Normal'}")

    # Plot histogram + KDE
    plt.figure(figsize=(6, 4))
    sns.histplot(wave_data, kde=True, bins=40, color='teal')
    plt.title(f'Distribution of {wave} (Shapiro p={p_shapiro:.4f})')
    plt.xlabel(f'{wave} value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
