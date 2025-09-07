#random forest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split



# random simulated EEG for testing
n_samples = 120
n_channels = 7
n_timepoints = 179
np.random.seed(42)
X = np.random.rand(n_samples, n_channels, n_timepoints)
Y = np.random.randint(35, 67, size=(n_samples,))  # age between 35 and 67

# Reshape X to 2D: (n_samples, n_channels * n_timepoints)
X_flat = X.reshape(X.shape[0], -1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, Y, test_size=0.1, random_state=42
)

# Fit Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.2f} years")
print(f"Test R²: {r2:.3f}")
