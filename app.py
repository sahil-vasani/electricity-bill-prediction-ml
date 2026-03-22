import streamlit as st
import pandas as pd
from src.predict import build_features, predict, load_model

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Electricity Bill Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS (UNCHANGED)
# ─────────────────────────────────────────────
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background-color: transparent;
}
.block-container {
    background: rgba(16, 37, 71, 0.55);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2rem 3rem;
    margin-top: 2rem;
    border: 1px solid rgba(255,255,255,0.15);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: rgba(16, 37, 71, 0.45) !important;
    backdrop-filter: blur(18px);
}
button[kind="primary"] {
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: #0f172a !important;
    font-weight: 800;
}
[data-testid="stMetricValue"] {
    font-size: 2.5rem;
    color: #92FE9D;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model("models/electricity_model.pkl")

saved = get_model()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

STATE_LIST = [
    "Gujarat","Maharashtra","Delhi","Karnataka",
    "Tamil Nadu","Rajasthan","Uttar Pradesh",
    "West Bengal","Madhya Pradesh","Telangana"
]

SEASON_INFO = {
    1:("❄️ Winter","Low electricity usage"),
    5:("🔥 Peak Summer","Highest usage"),
    6:("🌧️ Monsoon","Moderate usage"),
}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("⚡ Electricity Bill Predictor")
st.markdown("Estimate your **monthly electricity bill** based on your daily usage.")
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📅 Settings")

    state = st.selectbox("🏙️ Select State", STATE_LIST)

    month = st.selectbox(
        "📅 Month",
        list(range(1,13)),
        format_func=lambda x: MONTH_NAMES[x-1],
        index=4
    )

# ─────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────
st.subheader("🔌 Daily Appliance Usage")

col1, col2, col3 = st.columns(3)

with col1:
    fan = st.slider("🌀 Fan", 0, 24, 12)
    tv = st.slider("📺 TV", 0, 24, 5)

with col2:
    fridge = st.slider("🧊 Refrigerator", 0, 24, 22)
    monitor = st.slider("🖥️ Computer", 0, 24, 2)

with col3:
    ac = st.slider("❄️ AC", 0, 24, 0)
    motor = st.slider("💧 Motor", 0, 8, 1)

st.markdown("---")

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if st.button("🚀 Predict Electricity Bill", use_container_width=True):

    feature_df, physics_kwh, phys_bill = build_features(
        fan=fan,
        refrigerator=fridge,
        air_conditioner=ac,
        television=tv,
        monitor=monitor,
        motor_pump=motor,
        month=month,
        state=state
    )

    predicted_bill = predict(saved, feature_df, phys_bill)

    st.subheader("🔮 Estimated Bill")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("💰 Monthly Cost", f"₹{predicted_bill:.0f}")

    with c2:
        st.metric("⚡ Units Used", f"{physics_kwh:.0f} kWh")

    st.markdown("### 📈 Expected Range")
    st.write(f"₹{predicted_bill*0.9:.0f} — ₹{predicted_bill*1.1:.0f}")

    # ─────────────────────────────
    # Tips
    # ─────────────────────────────
    st.markdown("---")
    st.subheader("💡 Energy Tips")

    if ac > 6:
        st.info("❄️ Reduce AC usage or set to 24°C.")

    if predicted_bill > 5000:
        st.info("☀️ Consider solar panels.")

    if predicted_bill < 1500:
        st.success("✅ Efficient usage!")