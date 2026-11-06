from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import os

app = FastAPI(title="Hotel Booking Inference API", version="1.0.0")
MODEL_PATH = "models/random_forest.pkl"

class BookingFeatures(BaseModel):
    lead_time: int = Field(..., ge=0, example=34)
    arrival_date_year: int = Field(..., example=2026)
    arrival_date_week_number: int = Field(..., ge=1, le=53, example=27)
    arrival_date_day_of_month: int = Field(..., ge=1, le=31, example=1)
    stays_in_weekend_nights: int = Field(..., ge=0, example=0)
    stays_in_week_nights: int = Field(..., ge=0, example=2)
    adults: int = Field(..., ge=1, example=2)
    adr: float = Field(..., ge=0.0, example=98.5)

@app.get("/health")
def health_check():
    loaded = os.path.exists(MODEL_PATH)
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded}

@app.get("/metadata")
def get_metadata():
    return {
        "model_name": "Hotel Booking Random Forest Classifier",
        "framework": "scikit-learn",
        "target": "is_canceled"
    }

@app.post("/predict")
def predict_single(booking: BookingFeatures):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")
    
    model = joblib.load(MODEL_PATH)
    base_df = pd.read_csv("data/raw/hotel_bookings.csv").drop(columns=["is_canceled"], errors="ignore").head(1).copy()
    
    # Feature engineering matching training data
    base_df["total_nights"] = base_df["stays_in_weekend_nights"] + base_df["stays_in_week_nights"]
    base_df["total_guests"] = base_df["adults"] + base_df["children"].fillna(0) + base_df["babies"].fillna(0)
    base_df["is_family"] = ((base_df["children"] > 0) | (base_df["babies"] > 0)).astype(int)
    base_df["total_revenue"] = base_df["total_nights"] * base_df["adr"]
    
    # Inject API input values
    for key, val in booking.model_dump().items():
        if key in base_df.columns:
            base_df.loc[0, key] = val
            
    prediction = int(model.predict(base_df)[0])
    prob = float(model.predict_proba(base_df)[0][1]) if hasattr(model, "predict_proba") else None
    return {"prediction": prediction, "is_canceled": bool(prediction), "cancellation_probability": prob}