import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

DRIFT_THRESHOLD_PVALUE = 0.05
DATA_PATH = "data/raw/hotel_bookings.csv"
MODEL_PATH = "models/random_forest.pkl"

def check_data_drift(reference_df: pd.DataFrame, incoming_df: pd.DataFrame, features: list):
    """Detects statistical feature drift using two-sample KS test."""
    drift_status = {}
    any_drift = False
    for col in features:
        stat, p_value = ks_2samp(reference_df[col].dropna(), incoming_df[col].dropna())
        is_drifted = p_value < DRIFT_THRESHOLD_PVALUE
        if is_drifted:
            any_drift = True
        drift_status[col] = {
            "ks_stat": round(stat, 4),
            "p_value": round(p_value, 5),
            "drift_detected": is_drifted
        }
    return any_drift, drift_status

def retrain_model(data_path: str, model_output_path: str):
    """Retrains the model upon detected drift."""
    print("[RETRAIN] Statistical drift detected above threshold. Retraining model...")
    df = pd.read_csv(data_path)
    
    numeric_features = [
        "lead_time", "arrival_date_year", "arrival_date_week_number",
        "stays_in_weekend_nights", "stays_in_week_nights", "adults", "adr"
    ]
    X = df[numeric_features].fillna(0)
    y = df["is_canceled"]

    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"[SUCCESS] Retrained model saved successfully to {model_output_path}")

if __name__ == "__main__":
    base_df = pd.read_csv(DATA_PATH)

    # Simulate incoming drifted distribution
    simulated_production = base_df.copy()
    simulated_production["lead_time"] = simulated_production["lead_time"] * 2.5
    simulated_production["adr"] = simulated_production["adr"] + 50.0

    print("Running Drift Detection Analysis...")
    drift_flag, report = check_data_drift(base_df, simulated_production, ["lead_time", "adr"])
    for feat, met in report.items():
        print(f"Feature: {feat} | p-value: {met['p_value']} | Drift: {met['drift_detected']}")

    if drift_flag:
        retrain_model(DATA_PATH, MODEL_PATH)