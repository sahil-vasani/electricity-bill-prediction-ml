import joblib
import numpy as np
import pandas as pd
from src.slab_rates import calculate_slab_bill


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(path="models/electricity_model.pkl"):
    saved = joblib.load(path)

    print(
        f"✅ Model Loaded: {saved['model_name']} | "
        f"Val R²={saved.get('val_r2', 0):.4f} | "
        f"Test R²={saved.get('test_r2', 0):.4f}"
    )

    return saved


# ─────────────────────────────────────────────
# Compute physics (WITH SLAB)
# ─────────────────────────────────────────────
def compute_physics(fan, refrigerator, ac, tv, monitor, motor, state):

    physics_kwh = (
        fan * 70 +
        refrigerator * 150 +
        ac * 1500 +
        tv * 100 +
        monitor * 30 +
        motor * 750
    ) * 30 / 1000

    # 🔥 SLAB BILL
    physics_bill = calculate_slab_bill(physics_kwh, state)

    return physics_kwh, physics_bill


# ─────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────
def build_features(
    fan, refrigerator, air_conditioner,
    television, monitor, motor_pump,
    month, state
):

    # ───────── Physics ─────────
    physics_kwh, physics_bill = compute_physics(
        fan, refrigerator, air_conditioner,
        television, monitor, motor_pump,
        state
    )

    # ───────── AC ─────────
    ac_kwh = air_conditioner * 1500 * 30 / 1000

    # ───────── Temperature ─────────
    def approx_temp(month):
        if month in [12, 1, 2]:
            return 20
        elif month in [3, 4]:
            return 30
        elif month in [5, 6]:
            return 38
        elif month in [7, 8, 9]:
            return 32
        else:
            return 28

    temp = approx_temp(month)
    cooling_degree_days = max(temp - 24, 0)

    # ───────── Features ─────────
    temp_ac_interaction = ac_kwh * cooling_degree_days
    usage_intensity = physics_kwh / (fan + air_conditioner + 1)

    row = {
        'physics_bill': physics_bill,
        'ac_kwh': ac_kwh,
        'cooling_degree_days': cooling_degree_days,
        'temp_ac_interaction': temp_ac_interaction,
        'usage_intensity': usage_intensity
        
    }

    feature_df = pd.DataFrame([row])

    return feature_df, physics_kwh, physics_bill


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(saved, feature_df, physics_bill):
    
    model = saved['model']
    features = saved['features']

    # 🔥 FIX: Align features with model
    for col in features:
        if col not in feature_df.columns:
            feature_df[col] = 0  # fill missing

    feature_df = feature_df[features]  # ensure order

    # Prediction
    pred_ratio = float(model.predict(feature_df)[0])

    # Calibration
    pred_ratio = pred_ratio * 0.98

    # Safety clip
    pred_ratio = max(0.85, min(pred_ratio, 1.3))

    final_bill = physics_bill * pred_ratio

    # AC constraint
    ac_kwh = feature_df['ac_kwh'].values[0]
    if ac_kwh < 50:
        final_bill = min(final_bill, physics_bill * 1.15)

    # Global safety
    final_bill = max(final_bill, physics_bill * 0.5)
    final_bill = min(final_bill, physics_bill * 2.0)

    return round(final_bill, 2)