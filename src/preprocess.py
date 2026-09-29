import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/hotel_bookings.csv"
PROCESSED_DIR = "data/processed"

def run_preprocessing():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Loading raw data from", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Missing values
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    
    # 2. Prevent target leakage & drop unusable columns
    df = df.drop(columns=[
        "company", "agent", "reservation_status", 
        "reservation_status_date", "booking_changes"
    ])
    
    # 3. Filter invalid records
    df = df[df["adults"] + df["children"] + df["babies"] > 0]
    
    # 4. Feature Engineering
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["total_revenue"] = df["adr"] * df["total_nights"]
    
    # 5. Stratified Train-Test split (80/20)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["is_canceled"]
    )
    
    # Save processed datasets
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # 6. Save metadata (Experiment 1 requirement)
    metadata = {
        "dataset_name": "Hotel Booking Demand",
        "raw_shape": list(df.shape),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": [c for c in df.columns if c != "is_canceled"],
        "class_distribution": df["is_canceled"].value_counts(normalize=True).to_dict()
    }
    with open(os.path.join(PROCESSED_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Preprocessed train.csv, test.csv, and dataset_metadata.json created.")

if __name__ == "__main__":
    run_preprocessing()
