from src.load_data import load_data
from src.preprocess import preprocess_data
from src.feature_engineering import run_feature_engineering
from src.train_model import train_model


def main():

    print("\n🚀 ELECTRICITY BILL PREDICTION PIPELINE STARTED\n")

    # ── 1. Load ─────────────────────────────────────────────
    df = load_data("data/electricity_bill_dataset.csv")

    # ── 2. Preprocess ───────────────────────────────────────
    df = preprocess_data(df)

    # ── 3. Feature Engineering ─────────────────────────────
    df = run_feature_engineering(df)

    # ── 4. Save processed data ─────────────────────────────
    df.to_csv("data/processed.csv", index=False)
    print(f"\n✅ Processed data saved → data/processed.csv ({len(df)} rows)")

    # ── 5. Strong sanity check (IMPORTANT) ─────────────────
    print("\n📊 DATASET QUALITY CHECK")

    print(f"Bill Range: ₹{df['target_bill'].min():.0f} – ₹{df['target_bill'].max():.0f}")
    print(f"Physics Avg: ₹{df['physics_bill'].mean():.0f}")
    print(f"Actual Avg:  ₹{df['target_bill'].mean():.0f}")

    gap = abs(df['physics_bill'].mean() - df['target_bill'].mean()) / df['target_bill'].mean()

    print(f"Gap: {gap:.2f}")

    if gap > 0.25:
        print("⚠️ Dataset still noisy — results may be unstable")
    else:
        print("✅ Dataset looks consistent — good for training")

    # Residual stats
    print("\n📊 Ratio Stats:")
    print(f"Mean ratio: {df['ratio'].mean():.3f}")
    print(f"Std ratio:  {df['ratio'].std():.3f}")
    print(f"Min ratio:  {df['ratio'].min():.3f}")
    print(f"Max ratio:  {df['ratio'].max():.3f}")
    # ── 6. Train ───────────────────────────────────────────
    train_model(df)

    print("\n✅ PIPELINE COMPLETE — model ready 🚀")


if __name__ == "__main__":
    main()