import os
import joblib
import pandas as pd
import pytest

DATA_PATH = "data/raw/hotel_bookings.csv"
MODEL_PATH = "models/random_forest.pkl"

def test_dataset_exists():
    """Verify raw dataset exists in data/raw/."""
    assert os.path.exists(DATA_PATH), f"Error: Dataset file not found at {DATA_PATH}"

def test_model_artifact_load():
    """Verify saved model artifact exists and can predict."""
    assert os.path.exists(MODEL_PATH), f"Error: Model file {MODEL_PATH} not found."
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Error: Model missing predict() method."

def test_data_shape_and_columns():
    """Verify dataset is non-empty and has target column."""
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Error: Dataset is empty."
    assert "is_canceled" in df.columns, "Error: Target column 'is_canceled' missing."
