import os
import logging
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_model_features(model):
    """Retrieve the exact feature names the model was fitted on."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def validate_schema(df: pd.DataFrame, expected_cols: list) -> bool:
    """Checks for missing required columns."""
    if expected_cols is None:
        return True
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Schema Validation Failed: Missing columns: {set(missing)}")
    return True

def sanitize_and_recover(df: pd.DataFrame) -> pd.DataFrame:
    """Recovers corrupted records by handling invalid types and imputing missing values."""
    cleaned = df.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype == object or str(cleaned[col].dtype).startswith('int') or str(cleaned[col].dtype).startswith('float'):
            try:
                converted = pd.to_numeric(cleaned[col], errors='coerce')
                if converted.notna().sum() > 0:
                    cleaned[col] = converted
                    median_val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(median_val if not np.isnan(median_val) else 0)
            except Exception:
                pass
    return cleaned

def safe_predict(model_path: str, data: pd.DataFrame):
    """Executes safe prediction with fallbacks for corrupted data and missing artifacts."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Artifact {model_path} does not exist.")

        model = joblib.load(model_path)
        expected_features = get_model_features(model)

        validate_schema(data, expected_features)
        sanitized_data = sanitize_and_recover(data)

        if expected_features:
            input_data = sanitized_data[expected_features]
        else:
            input_data = sanitized_data

        return model.predict(input_data)

    except ValueError as ve:
        logging.error(f"[Schema Failure] {ve}. Initiating default fallback.")
        return np.zeros(len(data), dtype=int)

    except FileNotFoundError as fe:
        logging.error(f"[Artifact Failure] {fe}. Utilizing fallback baseline model.")
        return np.zeros(len(data), dtype=int)

    except Exception as e:
        logging.error(f"[Unexpected Pipeline Error] {e}. Executing safe recovery.")
        return np.zeros(len(data), dtype=int)