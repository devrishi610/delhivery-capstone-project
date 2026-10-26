import pandas as pd
import numpy as np

# Pinned constants from project brief
PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend",
    "dispatch_hour", "hub_tier", "hub_daily_capacity", "label_print_batch",
    "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no"
]

# ----------------------------------------------------------------
# TASK 2: clean_shipments(raw, medians=None)
# Takes a raw shipments DataFrame and returns (clean_df, medians)
# ----------------------------------------------------------------

def clean_shipments(raw, medians=None):
    """Clean raw shipments DataFrame and impute missing numeric values using medians."""
    df = raw.copy()

    # Step (i): Drop exact duplicate rows
    df = df.drop_duplicates().copy()

    # Step (ii): service_type - strip leading/trailing spaces and convert to title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # Step (iii): payment_mode - replace missing values with placeholder "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step (iv): declared_value - remove rupee (₹) sign and thousands commas,
    # convert to numeric, store in declared_value_inr, and drop original text column
    if "declared_value" in df.columns:
        cleaned_val = (
            df["declared_value"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        cleaned_val = cleaned_val.replace({"nan": np.nan, "None": np.nan, "": np.nan})
        df["declared_value_inr"] = pd.to_numeric(cleaned_val, errors="coerce")
        df = df.drop(columns=["declared_value"])

    # Step (v): dispatch_time - convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step (vi): parcel_weight_kg - treat any value above 100 as a scanner-fault sentinel (missing)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Step (vii): Fill missing distance_km, parcel_weight_kg, and declared_value_inr with column medians
    # Compute if median is None, else use passed dictioary
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median())
        }

    for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]:
        df[col] = df[col].fillna(medians[col])

    return df, medians

# -----------------------------------------------------------------------
# TASK 3: build_features(clean, hubs)
# Left merges hub columns and builds new features without delivery_hours
# -----------------------------------------------------------------------

def build_features(clean, hubs):
    """Attach hub metadata via left merge and construct model features without using delivery_hours."""
    df = clean.copy()

    # Attach hub columns by a left merge on hub_id
    df = df.merge(hubs, on="hub_id", how="left")

    # Construct engineered features

    # 1. promised_hours(from PROMISED_HOURS, integer)
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)

    # 2. is_cod (1 when payment_mode is COD, else 0. Unknown gives 0)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # 3. is_weekend (1 when dispatch day is Saturday or Sunday, else 0)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    # 4. dispatch_hour (hour of dispatch timestamp as integer, 6 to 21)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    # 5. log_declared_value (natural log of declared_value_inr)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
