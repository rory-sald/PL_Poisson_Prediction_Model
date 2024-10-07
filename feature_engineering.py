# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from backtest_module import full_data  # Importing the full dataset

import pandas as pd

# Example: Calculate lagged and moving average features
lags = [1, 2, 3]  # Define the number of past games to look at
metrics = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY']  # Relevant metrics

# Create lagged features
for metric in metrics:
    for lag in lags:
        # Lagged values for home team performance
        full_data[f'{metric}_Lag{lag}'] = full_data.groupby('HomeTeam')[metric].shift(lag)
        # Lagged values for away team performance
        full_data[f'{metric}_Lag{lag}_Away'] = full_data.groupby('AwayTeam')[metric].shift(lag)

# Create rolling average features
for metric in metrics:
    full_data[f'Avg_{metric}_Last3'] = full_data.groupby('HomeTeam')[metric].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    full_data[f'Avg_{metric}_Last3_Away'] = full_data.groupby('AwayTeam')[metric].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Step 1: Preprocess the data
# Select numerical columns only (exclude columns with string values such as team names)
numerical_data = full_data.select_dtypes(include=[np.number])

# List of columns to drop
columns_to_drop = [
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',  # Outcome and Half-time labels
    'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Referee',  # Match-specific and team names
    'B365H', 'B365D', 'B365A', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',  # Odds and betting data
    'B365>2.5', 'B365<2.5', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',  # Additional odds metrics
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'  # In-game performance stats
]

# Drop these columns from the numerical data
numerical_data = numerical_data.drop(columns=columns_to_drop, errors='ignore')

# Step 2: Handle Missing Values
# Impute missing values with the median for each column
imputer = SimpleImputer(strategy='median')
numerical_data_imputed = pd.DataFrame(imputer.fit_transform(numerical_data), columns=numerical_data.columns)

# Add target variable back for correlation
numerical_data_imputed['FTHG'] = full_data['FTHG']

# Step 3: Identify correlations with the target variable (e.g., home goals)
# Calculate correlation matrix
corr_matrix = numerical_data_imputed.corr()

# Get absolute correlation values with the target to identify most relevant features
corr_with_target = corr_matrix['FTHG'].abs().sort_values(ascending=False)
print("Top Correlated Features with Home Goals:")
print(corr_with_target.head(10))  # Display top 10 features

# Step 4: Feature Importance using Random Forest
# Define target and features
X = numerical_data_imputed.drop(columns=['FTHG'])
y = numerical_data_imputed['FTHG']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Get feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
print(feature_importances.head(10))

# Step 5: Recursive Feature Elimination (RFE) with Lasso
# Fit LassoCV to find the optimal alpha parameter
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
rfe = RFE(estimator=lasso, n_features_to_select=10)
rfe.fit(X_scaled, y)

# Identify selected features
selected_features = X.columns[rfe.support_]
print("\nTop Features Selected by RFE with Lasso:")
print(selected_features)
