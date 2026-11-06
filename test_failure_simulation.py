import pandas as pd
import numpy as np
from src.failure_handler import safe_predict

# 1. Load sample rows from the raw dataset
raw_data = pd.read_csv("data/raw/hotel_bookings.csv").drop(columns=["is_canceled"], errors="ignore")
corrupt_payload = raw_data.head(2).copy()

# Add the 4 engineered features created during earlier experiments
corrupt_payload["total_nights"] = corrupt_payload["stays_in_weekend_nights"] + corrupt_payload["stays_in_week_nights"]
corrupt_payload["total_guests"] = corrupt_payload["adults"] + corrupt_payload["children"].fillna(0) + corrupt_payload["babies"].fillna(0)
corrupt_payload["is_family"] = ((corrupt_payload["children"] > 0) | (corrupt_payload["babies"] > 0)).astype(int)
corrupt_payload["total_revenue"] = corrupt_payload["total_nights"] * corrupt_payload["adr"]

# Cast columns to object dtype so pandas allows inserting corrupt strings
corrupt_payload["lead_time"] = corrupt_payload["lead_time"].astype(object)
corrupt_payload["stays_in_week_nights"] = corrupt_payload["stays_in_week_nights"].astype(object)
corrupt_payload["adr"] = corrupt_payload["adr"].astype(object)

# 2. Inject corrupted values (invalid strings and NaN)
corrupt_payload.loc[0, "lead_time"] = "corrupt_lead_time_string"
corrupt_payload.loc[1, "adr"] = np.nan
corrupt_payload.loc[0, "stays_in_week_nights"] = "not_a_number"

print("\n--- Test 1: Corrupted Data Recovery ---")
output_1 = safe_predict("models/random_forest.pkl", corrupt_payload)
print("Output Predictions (Recovered):", output_1)

print("\n--- Test 2: Missing Model Artifact Fallback ---")
output_2 = safe_predict("models/non_existent_model.pkl", corrupt_payload)
print("Output Predictions (Fallback):", output_2)