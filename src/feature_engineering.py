import pandas as pd
import numpy as np


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering focused on real-world electricity usage patterns.
    Designed for residual learning (actual - physics).
    """

    df = df.copy()

    print("\n⚙️ Running feature engineering...")

    # ─────────────────────────────────────────────
    # 1. Cooling Demand (MOST IMPORTANT AFTER PHYSICS)
    # ─────────────────────────────────────────────
    # AC usage increases when temperature > 24°C
    # Since we don't have exact temperature dataset,
    # we approximate using month (Indian climate logic)

    def approx_temp(month):
        # Rough India temperature approximation
        if month in [12, 1, 2]:
            return 20   # winter
        elif month in [3, 4]:
            return 30   # warm
        elif month in [5, 6]:
            return 38   # peak summer
        elif month in [7, 8, 9]:
            return 32   # monsoon
        else:
            return 28   # moderate

    df['temperature'] = df['Month'].apply(approx_temp)

    # Cooling Degree Days (key feature in energy models)
    df['cooling_degree_days'] = (df['temperature'] - 24).clip(lower=0)

    # ─────────────────────────────────────────────
    # 2. Seasonal Effect
    # ─────────────────────────────────────────────
    def get_season(m):
        if m in [12, 1, 2]:
            return 0  # winter
        elif m in [3, 4, 5]:
            return 1  # summer
        elif m in [6, 7, 8, 9]:
            return 2  # monsoon
        else:
            return 3  # post-monsoon

    df['season'] = df['Month'].apply(get_season)

    # ─────────────────────────────────────────────
    # 3. Cyclic Month Encoding (VERY IMPORTANT)
    # ─────────────────────────────────────────────
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # ─────────────────────────────────────────────
    # 4. AC Dominance Feature (KEY SIGNAL)
    # ─────────────────────────────────────────────
    df['ac_kwh'] = df['AirConditioner'] * 1500 * 30 / 1000

    df['ac_ratio'] = df['ac_kwh'] / df['physics_kwh'].clip(lower=1)

    # If AC dominates usage → higher bill variability
    df['high_ac_usage'] = (df['ac_ratio'] > 0.4).astype(int)

    # ─────────────────────────────────────────────
    # 5. Total Appliance Intensity
    # ─────────────────────────────────────────────
    df['total_hours'] = (
        df['Fan'] + df['Refrigerator'] + df['AirConditioner'] +
        df['Television'] + df['Monitor'] + df['MotorPump']
    )
    
    df['avg_hours_per_device'] = df['total_hours'] / 6

    # ─────────────────────────────────────────────
    # 6. Tariff Impact Feature
    # ─────────────────────────────────────────────
    # Higher tariff → slightly nonlinear bill behavior
    df['log_tariff'] = np.log1p(df['TariffRate'])

    # ─────────────────────────────────────────────
    # 7. Interaction (VERY LIMITED — avoid overfitting)
    # ─────────────────────────────────────────────
    df['ac_temp_interaction'] = df['ac_kwh'] * df['cooling_degree_days'] 
    df['temp_ac_interaction'] = df['ac_kwh'] * df['cooling_degree_days']
    df['usage_intensity'] = df['physics_kwh'] / (df['Fan'] + df['AirConditioner'] + 1)
    # ─────────────────────────────────────────────
    # 8. Keep dataset clean
    # ─────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print(f"✅ Feature engineering complete → {df.shape[1]} columns")

    return df