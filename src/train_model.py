import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


# ✅ FINAL FEATURES (STRONG + STABLE)
FEATURES = [
    'physics_bill',
    'ac_kwh',
    'cooling_degree_days',
    'temp_ac_interaction',
    'usage_intensity',
]

TARGET = 'ratio'


# ─────────────────────────────
# Pipeline
# ─────────────────────────────
def make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])


# ─────────────────────────────
# Evaluation
# ─────────────────────────────
def evaluate(name, model, X, y, physics):
    pred_ratio = model.predict(X)
    pred_bill = physics * pred_ratio
    actual = physics * y

    r2 = r2_score(actual, pred_bill)
    rmse = np.sqrt(mean_squared_error(actual, pred_bill))
    mae = mean_absolute_error(actual, pred_bill)

    print(f"{name:<20} | R²={r2:.4f} | RMSE=₹{rmse:.0f} | MAE=₹{mae:.0f}")
    return r2


# ─────────────────────────────
# MAIN TRAIN FUNCTION
# ─────────────────────────────
def train_model(df: pd.DataFrame):

    print("\n🚀 TRAINING (TRAIN / VAL / TEST SPLIT)\n")

    df = df.copy()
  
    X = df[FEATURES]
    y = df[TARGET]
    physics = df['physics_bill']

    # ─────────────────────────────
    # Split: 70 / 15 / 15
    # ─────────────────────────────
    X_train, X_temp, y_train, y_temp, phys_train, phys_temp = train_test_split(
        X, y, physics, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test, phys_val, phys_test = train_test_split(
        X_temp, y_temp, phys_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ─────────────────────────────
    # BASE MODELS + SEARCH
    # ─────────────────────────────
    model_configs = {

        "Ridge": (
            make_pipeline(Ridge()),
            {
                "model__alpha": np.logspace(-2, 2, 10)
            }
        ),

        "RandomForest": (
            make_pipeline(RandomForestRegressor(random_state=42)),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
                "model__min_samples_leaf": [10, 20]
            }
        ),

        "XGBoost": (
            make_pipeline(XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)),
            {
                "model__n_estimators": [100, 150],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.6, 0.8],
                "model__colsample_bytree": [0.6, 0.8],
                "model__reg_alpha": [1, 2],
                "model__reg_lambda": [5, 10],
                "model__min_child_weight": [10, 20]
            }
        )
    }

    trained_models = {}

    # ─────────────────────────────
    # Train + Tune each model
    # ─────────────────────────────
    for name, (pipe, params) in model_configs.items():

        print(f"\n🔍 Tuning {name}...")

        search = RandomizedSearchCV(
            pipe,
            params,
            n_iter=8,
            scoring='r2',
            cv=3,
            random_state=42,
            n_jobs=1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        print(f"Best params: {search.best_params_}")

        evaluate(name + " (VAL)", best_model, X_val, y_val, phys_val)

        trained_models[name] = best_model

    # ─────────────────────────────
    # 🔥 STACKING ENSEMBLE (FINAL MODEL)
    # ─────────────────────────────
    print("\n🚀 Training Stacking Ensemble...")

    stack_model = StackingRegressor(
        estimators=[
            ('ridge', trained_models['Ridge']),
            ('rf', trained_models['RandomForest']),
            ('xgb', trained_models['XGBoost'])
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=1
    )

    stack_model.fit(X_train, y_train)

    val_r2 = evaluate("STACK (VAL)", stack_model, X_val, y_val, phys_val)
    test_r2 = evaluate("STACK (TEST)", stack_model, X_test, y_test, phys_test)

    print("\n🏆 FINAL MODEL: STACKING ENSEMBLE")
    print(f"Validation R²: {val_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # ─────────────────────────────
    # SAVE
    # ─────────────────────────────
    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": stack_model,
        "features": FEATURES,
        "model_name": "Stacking Ensemble",
        "val_r2": val_r2,
        "test_r2": test_r2
    }, "models/electricity_model.pkl")

    print("\n✅ Model saved → STACKING MODEL")

    return stack_model