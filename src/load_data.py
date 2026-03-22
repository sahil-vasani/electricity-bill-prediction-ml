import pandas as pd
import os


REQUIRED_COLUMNS = [
    'Fan', 'Refrigerator', 'AirConditioner', 'Television',
    'Monitor', 'MotorPump', 'Month', 'City',
    'MonthlyHours', 'TariffRate', 'ElectricityBill'
]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Robust data loader with validation and sanity checks.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")

    print(f"\n✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # ─────────────────────────────────────────────
    # Column validation
    # ─────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns: {missing}")

    print("✅ All required columns present")

    # ─────────────────────────────────────────────
    # Remove duplicates
    # ─────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"🧹 Removed {before - len(df)} duplicate rows")

    return df