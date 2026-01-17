import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------------------------------------
# 1. LOAD THE REAL DATA
# ---------------------------------------------------------
try:
    df = pd.read_csv('golf_stats_history.csv')
    print(f"Successfully loaded {len(df)} player records.")
except FileNotFoundError:
    print("Error: 'golf_stats_history.csv' not found. Please run scraper.py first.")
    exit()

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
# We need to create a target variable that represents "DFS Fantasy Points".
# Since we have summary data, we will engineer a "Fantasy Potential Score" per round.
# Formula: Base 50pts + (Birdies * 3) + (Beating Par * 1.5)
# This approximates a good DraftKings/FanDuel round score.

# Note: SCORE is scoring average. 72 is par.
df['Fantasy_Proxy'] = 50 + (df['BIRDS'] * 3) + ((72 - df['SCORE']) * 1.5)

# Select the Features (The stats that predict the score)
# DDIS = Driving Distance
# DACC = Driving Accuracy
# GIR  = Greens in Regulation %
# PUTTS = Putting Average
# SAND = Sand Save %
feature_cols = ['DDIS', 'DACC', 'GIR', 'PUTTS', 'SAND']
target_col = 'Fantasy_Proxy'

# Clean data (Drop any rows where stats might be missing/zero)
df_clean = df[feature_cols + [target_col]].dropna()

# ---------------------------------------------------------
# 3. SPLIT DATA
# ---------------------------------------------------------
X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. TRAIN XGBOOST MODEL
# ---------------------------------------------------------
print("Training Model on Real Data...")

# XGBoost Regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,        # Number of boosting rounds
    learning_rate=0.05,      # Step size shrinkage
    max_depth=5,             # Depth of trees
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. EVALUATE
# ---------------------------------------------------------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Performance ---")
print(f"Mean Absolute Error: {mae:.2f} DFS Points")
print(f"R-Squared (Accuracy): {r2:.2f}")

# Show Feature Importance (What stats matter most?)
print("\n--- Key Stat Drivers ---")
importance = model.feature_importances_
for i, col in enumerate(feature_cols):
    print(f"{col}: {importance[i]:.4f}")

# ---------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------
joblib.dump(model, 'golf_model.pkl')
print("\nModel saved as 'golf_model.pkl'")