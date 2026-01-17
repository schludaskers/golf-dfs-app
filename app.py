import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="GreenReader DFS",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Trendy" Look (Dark Mode Compatible)
st.markdown("""
<style>
    /* Card Styling */
    .stContainer {
        border-radius: 15px !important;
        border: 1px solid #333;
        background-color: #1E1E1E;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #4CAF50; /* Golf Green */
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }

    /* Custom Badge for Value */
    .value-badge {
        background-color: #2ecc71;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS & LOADING
# ---------------------------------------------------------

# A. Load the ML Model
@st.cache_resource
def load_model():
    try:
        # Tries to load the model you trained in model_train.py
        return joblib.load('golf_model.pkl')
    except:
        return None


# B. Player Headshot Mapper
def get_headshot(player_name):
    # Map common names to ESPN Player IDs for real photos
    id_map = {
        "Scottie Scheffler": "4604687",
        "Rory McIlroy": "3470",
        "Xander Schauffele": "10140",
        "Viktor Hovland": "4364873",
        "Ludvig Aberg": "4845369",
        "Max Homa": "8973",
        "Patrick Cantlay": "6007",
        "Collin Morikawa": "4405068",
        "Jordan Spieth": "5467",
        "Justin Thomas": "4848",
        "Hideki Matsuyama": "5860",
        "Keegan Bradley": "3615"
    }

    player_id = id_map.get(player_name)

    if player_id:
        return f"https://a.espncdn.com/combiner/i?img=/i/headshots/golf/players/full/{player_id}.png&w=350&h=254"
    else:
        # Generic Silhouette
        return "https://a.espncdn.com/combiner/i?img=/i/headshots/nophoto.png&w=350&h=254"


# C. Data Source (Hybrid: Mock vs Live)
def get_field_data(use_live_api=False, api_key=None):
    if use_live_api and api_key:
        # --- RAPID API PLACEHOLDER ---
        # If you subscribe to an API later, put the code here
        st.warning("Live API not fully configured. Using Mock Data.")
        pass

    # --- MOCK DATA GENERATOR (Realistic Stats) ---
    players = [
        'Scottie Scheffler', 'Rory McIlroy', 'Xander Schauffele', 'Viktor Hovland',
        'Ludvig Aberg', 'Max Homa', 'Patrick Cantlay', 'Collin Morikawa',
        'Jordan Spieth', 'Justin Thomas', 'Hideki Matsuyama', 'Keegan Bradley',
        'Tony Finau', 'Cameron Young', 'Sahith Theegala', 'Jason Day'
    ]

    data = []
    np.random.seed(42)  # Keeps data consistent on refresh

    for p in players:
        # Realistic ranges for the new stats columns
        salary = np.random.randint(7000, 12500)

        # DDIS: Driving Distance (Avg 290-320)
        ddis = round(np.random.uniform(290, 320), 1)

        # DACC: Driving Accuracy % (Avg 50-75)
        dacc = round(np.random.uniform(50, 75), 1)

        # GIR: Greens in Regulation % (Avg 60-75)
        gir = round(np.random.uniform(60, 75), 1)

        # PUTTS: Putts per GIR (Avg 1.65 - 1.80) - Lower is better
        putts = round(np.random.uniform(1.65, 1.80), 3)

        # SAND: Sand Save % (Avg 40-65)
        sand = round(np.random.uniform(40, 65), 1)

        data.append([p, salary, ddis, dacc, gir, putts, sand])

    return pd.DataFrame(data, columns=['Player', 'Salary', 'DDIS', 'DACC', 'GIR', 'PUTTS', 'SAND'])


# ---------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2413/2413074.png", width=80)
    st.title("Settings")

    st.markdown("### 💰 Budgeting")
    budget = st.number_input("Remaining Salary Cap", value=50000, step=100)

    st.info("Model trained on 2024-2025 Real Data.")
    st.divider()
    st.caption("v2.0.0 | Updated for Real Stats")

# ---------------------------------------------------------
# 4. MAIN APP LOGIC
# ---------------------------------------------------------
st.title("⛳ GreenReader DFS")
st.markdown("**Tournament Prediction Model** | Powered by XGBoost & Real 2024 Stats")

# Load Model
model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run 'model_train.py' first to generate 'golf_model.pkl'.")
    st.stop()

# Get Data
df = get_field_data()

# Predict
# We pass the EXACT columns the model was trained on
features = df[['DDIS', 'DACC', 'GIR', 'PUTTS', 'SAND']]
df['Proj_Pts'] = model.predict(features)

# Calculate 'Value' (Points per $1,000 salary)
df['Value'] = df['Proj_Pts'] / (df['Salary'] / 1000)

# Sort by Projection
df = df.sort_values(by='Proj_Pts', ascending=False).reset_index(drop=True)

# ---------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------

# --- A. Top 3 "Hero" Cards ---
st.subheader("🔥 Optimal Plays")
col1, col2, col3 = st.columns(3)

top_picks = df.head(3)

# Loop through the top 3
for idx, col in enumerate([col1, col2, col3]):
    if idx < len(top_picks):
        player = top_picks.iloc[idx]

        with col:
            with st.container():
                # Header
                st.markdown(f"#### #{idx + 1} {player['Player']}")

                # Headshot
                st.image(get_headshot(player['Player']), use_container_width=True)

                # Key Stats Row
                c1, c2 = st.columns(2)
                c1.metric("Points", f"{player['Proj_Pts']:.1f}")
                c2.metric("Salary", f"${player['Salary']:,}")

                # Context Stat
                st.caption(f"Drive: {player['DDIS']}y | GIR: {player['GIR']}%")

                # Value Bar
                st.progress(min(player['Value'] / 10, 1.0), text=f"Value Rating: {player['Value']:.2f}x")

st.divider()

# --- B. Full Data Table ---
st.subheader("📋 Full Field Analysis")

# Setup the dataframe for display
# Renaming columns for cleaner UI reading
display_df = df[['Player', 'Salary', 'Proj_Pts', 'Value', 'DDIS', 'GIR', 'PUTTS']].copy()

st.dataframe(
    display_df,
    column_config={
        "Player": "Golfer",
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "Proj_Pts": st.column_config.ProgressColumn("Proj. Points", format="%.1f", min_value=0, max_value=150),
        "Value": st.column_config.NumberColumn("Value (x)", format="%.2f"),
        "DDIS": st.column_config.NumberColumn("Driving Dist", format="%.1f yds"),
        "GIR": st.column_config.NumberColumn("GIR %", format="%.1f%%"),
        "PUTTS": st.column_config.NumberColumn("Putts/GIR", format="%.2f"),
    },
    use_container_width=True,
    height=600,
    hide_index=True
)