import pandas as pd
import numpy as np


# Realistic appliance wattage (India average)
APPLIANCE_WATTS = {
    'Fan': 70,
    'Refrigerator': 150,
    'AirConditioner': 1500,
    'Television': 100,
    'Monitor': 30,
    'MotorPump': 750,
}


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("\n🔧 Starting preprocessing...")

    # ─────────────────────────────────────────────
    # 1. Convert numeric safely
    # ─────────────────────────────────────────────
    num_cols = [
        'Fan', 'Refrigerator', 'AirConditioner',
        'Television', 'Monitor', 'MotorPump',
        'Month', 'TariffRate', 'ElectricityBill'
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    before = len(df)
    df = df.dropna(subset=num_cols)
    print(f"🧹 Removed {before - len(df)} null rows")

    # ─────────────────────────────────────────────
    # 2. Clean basic ranges (VERY IMPORTANT)
    # ─────────────────────────────────────────────
    df = df[
        (df['Month'] >= 1) & (df['Month'] <= 12) &
        (df['TariffRate'] >= 2) & (df['TariffRate'] <= 20) &
        (df['ElectricityBill'] > 0)
    ]

    # ─────────────────────────────────────────────
    # 3. Compute physics kWh (CORE LOGIC)
    # ─────────────────────────────────────────────
    df['physics_kwh'] = (
        df['Fan'] * 70 +
        df['Refrigerator'] * 150 +
        df['AirConditioner'] * 1500 +
        df['Television'] * 100 +
        df['Monitor'] * 30 +
        df['MotorPump'] * 750
    ) * 30 / 1000

    # ─────────────────────────────────────────────
    # 4. Compute physics bill
    # ─────────────────────────────────────────────
    df['physics_bill'] = df['physics_kwh'] * df['TariffRate']

    # ─────────────────────────────────────────────
    # 5. Remove unrealistic rows (MOST IMPORTANT)
    # ─────────────────────────────────────────────
    df['ratio'] = df['ElectricityBill'] / df['physics_bill'].clip(lower=1)

    before = len(df)

    # Keep only realistic rows
    df = df[
        (df['ratio'] >= 0.7) &   # not too low
        (df['ratio'] <= 1.5)     # not too high
    ]

    print(f"🚫 Removed {before - len(df)} inconsistent rows")

    # ─────────────────────────────────────────────
    # 6. Clip extreme bills
    # ─────────────────────────────────────────────
    q1 = df['ElectricityBill'].quantile(0.01)
    q99 = df['ElectricityBill'].quantile(0.99)

    df = df[(df['ElectricityBill'] >= q1) & (df['ElectricityBill'] <= q99)]

    # ─────────────────────────────────────────────
    # 7. Target
    # ─────────────────────────────────────────────
    df['target_bill'] = df['ElectricityBill']

    # ─────────────────────────────────────────────
    # 8. Residual (BEST APPROACH)
    # ─────────────────────────────────────────────
    # 🔥 NEW TARGET (RATIO)
    df['ratio'] = df['target_bill'] / df['physics_bill'].clip(lower=1)

    # Clip to remove noise
    df['ratio'] = df['ratio'].clip(0.85, 1.3)

    # ─────────────────────────────────────────────
    # 9. Simple feature (important)
    # ─────────────────────────────────────────────
    df['ac_kwh'] = df['AirConditioner'] * 1500 * 30 / 1000  
    # ─────────────────────────────────────────────
    # 10. Final cleanup
    # ─────────────────────────────────────────────
    df = df.reset_index(drop=True)

    print(f"✅ Final dataset: {len(df)} rows")

    # ─────────────────────────────────────────────
    # 11. Sanity check
    # ─────────────────────────────────────────────
    print("\n📊 SANITY CHECK")
    print(f"Physics avg: ₹{df['physics_bill'].mean():.0f}")
    print(f"Actual avg:  ₹{df['target_bill'].mean():.0f}")

    gap = abs(df['physics_bill'].mean() - df['target_bill'].mean()) / df['target_bill'].mean()

    if gap > 0.2:
        print(f"⚠️ WARNING: dataset still noisy (gap={gap:.2f})")
    else:
        print(f"✅ Dataset is consistent (gap={gap:.2f})")

    return df