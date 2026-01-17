import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib


# ---------------------------------------------------------
# 1. GENERATE MOCK DATA (Replace this with your real CSV load)
# ---------------------------------------------------------
def generate_mock_data(n_rows=1000):
    np.random.seed(42)
    players = ['Scottie Scheffler', 'Rory McIlroy', 'Xander Schauffele',
               'Viktor Hovland', 'Ludvig Aberg', 'Max Homa', 'Patrick Cantlay',
               'Collin Morikawa', 'Jordan Spieth', 'Justin Thomas']

    data = []
    for _ in range(n_rows):
        player = np.random.choice(players)
        # Random stats logic to make it somewhat realistic
        sg_ott = np.random.normal(0.5, 0.8)  # Off the tee
        sg_app = np.random.normal(0.5, 0.9)  # Approach
        sg_putt = np.random.normal(0, 1.0)  # Putting
        course_difficulty = np.random.choice([0.8, 1.0, 1.2])  # Hard, Medium, Easy

        # Target: DFS Points (correlated with stats)
        # Base score + stats weight + randomness
        dfs_points = 60 + (sg_ott * 5) + (sg_app * 8) + (sg_putt * 3) + np.random.normal(0, 5)

        data.append([player, sg_ott, sg_app, sg_putt, course_difficulty, dfs_points])

    return pd.DataFrame(data, columns=['Player', 'SG_OTT', 'SG_APP', 'SG_PUTT', 'Course_Diff', 'DFS_Points'])


# LOAD DATA
print("Loading data...")
df = generate_mock_data(2000)
# In real life: df = pd.read_csv('pga_data_2024.csv')

# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
# We drop 'Player' for training because the model shouldn't memorize names,
# it should learn from the stats.
X = df[['SG_OTT', 'SG_APP', 'SG_PUTT', 'Course_Diff']]
y = df['DFS_Points']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. TRAIN XGBOOST MODEL
# ---------------------------------------------------------
print("Training Model...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model Mean Absolute Error: {mae:.2f} DFS Points")

# ---------------------------------------------------------
# 4. SAVE THE MODEL
# ---------------------------------------------------------
joblib.dump(model, 'golf_model.pkl')
print("Model saved as 'golf_model.pkl'")