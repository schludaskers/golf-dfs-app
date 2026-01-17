import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="GreenReader DFS",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Trendy" Look (Dark mode compatible)
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; color: #2E86C1; }
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; border-left: 5px solid #2ecc71; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('golf_model.pkl')


model = load_model()

# ---------------------------------------------------------
# SIDEBAR: USER INPUTS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2413/2413074.png", width=100)
    st.title("DFS Settings")

    st.markdown("### 🏟️ Course Conditions")
    difficulty = st.select_slider(
        "Course Difficulty",
        options=["Easy", "Medium", "Hard"],
        value="Medium"
    )
    diff_map = {"Easy": 1.2, "Medium": 1.0, "Hard": 0.8}

    st.markdown("### 📊 Weighting")
    st.info("Adjust how much recent form matters vs. long term history.")
    recency_bias = st.slider("Recency Bias %", 0, 100, 20)

# ---------------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------------
st.title("⛳ GreenReader: Golf DFS Predictor")
st.markdown("AI-Powered Projections for this week's tournament.")

# SIMULATE UPCOMING TOURNAMENT DATA (In real life, you fetch this via API)
current_field = pd.DataFrame({
    'Player': ['Scottie Scheffler', 'Rory McIlroy', 'Xander Schauffele', 'Viktor Hovland', 'Ludvig Aberg',
               'Max Homa', 'Patrick Cantlay', 'Collin Morikawa', 'Jordan Spieth', 'Justin Thomas'],
    'SG_OTT': [0.95, 0.92, 0.85, 0.80, 0.78, 0.70, 0.65, 0.60, 0.55, 0.50],
    'SG_APP': [1.10, 0.88, 0.95, 0.85, 0.82, 0.75, 0.70, 0.90, 0.60, 0.55],
    'SG_PUTT': [0.20, 0.30, 0.40, 0.25, 0.35, 0.50, 0.45, 0.30, 0.60, 0.10],
    'Salary': [11500, 11200, 10800, 10500, 9800, 9500, 9200, 9000, 8800, 8500]
})

# Add feature columns expected by the model
current_field['Course_Diff'] = diff_map[difficulty]

# PREDICT
features = current_field[['SG_OTT', 'SG_APP', 'SG_PUTT', 'Course_Diff']]
current_field['Proj_Pts'] = model.predict(features)
current_field['Value'] = current_field['Proj_Pts'] / (current_field['Salary'] / 1000)

# SORT BY PROJECTION
current_field = current_field.sort_values(by='Proj_Pts', ascending=False).reset_index(drop=True)

# ---------------------------------------------------------
# TOP 3 "PLAYER CARDS" (Trendy UI)
# ---------------------------------------------------------
st.subheader("🔥 Top 3 Optimal Plays")

cols = st.columns(3)
for i in range(3):
    player = current_field.iloc[i]
    with cols[i]:
        # Clean Container
        with st.container(border=True):
            st.markdown(f"#### #{i + 1} {player['Player']}")

            # Dynamic Headshot (Using ESPN public pattern or placeholder)
            # Note: This is a placeholder URL structure.
            st.image(f"https://a.espncdn.com/combiner/i?img=/i/headshots/golf/players/full/1234.png&w=350&h=254",
                     caption=f"${player['Salary']:,}", use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Proj Pts", f"{player['Proj_Pts']:.1f}")
            c2.metric("Value", f"{player['Value']:.2f}x")

            st.progress(player['Proj_Pts'] / 100, text="Confidence")

st.divider()

# ---------------------------------------------------------
# DETAILED DATAFRAME
# ---------------------------------------------------------
st.subheader("📋 Full Field Projections")

st.dataframe(
    current_field.style.background_gradient(subset=['Proj_Pts', 'Value'], cmap="Greens"),
    column_config={
        "Player": "Golfer",
        "Proj_Pts": st.column_config.NumberColumn("Points", format="%.1f"),
        "Salary": st.column_config.NumberColumn("DK Salary", format="$%d"),
        "Value": st.column_config.NumberColumn("Value (x)", format="%.2f"),
        "Course_Diff": None,  # Hide technical columns
        "SG_OTT": None,
        "SG_APP": None,
        "SG_PUTT": None
    },
    use_container_width=True,
    height=500
)