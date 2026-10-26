import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, PROMISED_HOURS, FEATURES

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ------------------------------------------------------------------------------
# TASK 4: Read data, call functions, add is_delayed & write modelling_table.csv
# ------------------------------------------------------------------------------
raw_df = pd.read_csv("data/raw_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")

# Call clean_shipments and build_features
clean_df, medians = clean_shipments(raw_df, medians=None)
model_df = build_features(clean_df, hubs_df)

# Add target is_delayed (1 when delivery_hours > promised_hours, else 0)
model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

# Save modelling table
model_df.to_csv("data/modelling_table.csv", index=False)


# ------------------------------------------------
# TASK 5: Checks file (outputs/part_a_checks.csv)
# ------------------------------------------------
# Measure "Before" counts after removing duplicates and before any filling
dedup_df = raw_df.drop_duplicates()

# Calculate quartiles and 1.5 * IQR bounds from unrounded quartiles
q1 = model_df["delivery_hours"].quantile(0.25)
q3 = model_df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = ((model_df["delivery_hours"] < lower_bound) | (model_df["delivery_hours"] > upper_bound)).sum()

checks_data = [
    ("raw_rows", len(raw_df)),
    ("duplicate_rows_removed", len(raw_df) - len(dedup_df)),
    ("rows_after_dedup", len(dedup_df)),
    ("missing_distance_km_before", dedup_df["distance_km"].isna().sum()),
    ("missing_parcel_weight_before", dedup_df["parcel_weight_kg"].isna().sum()),
    ("sentinel_weight_rows", (dedup_df["parcel_weight_kg"] > 100).sum()),
    ("missing_declared_value_before", dedup_df["declared_value"].isna().sum()),
    ("missing_payment_mode_before", dedup_df["payment_mode"].isna().sum()),
    ("service_type_spellings_before", dedup_df["service_type"].nunique()),
    ("service_type_spellings_after", model_df["service_type"].nunique()),
    ("median_distance_km", round(medians["distance_km"], 4)),
    ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
    ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
    ("rows_after_hub_merge", len(model_df)),
    ("nulls_in_features_and_target", model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    ("delayed_count", model_df["is_delayed"].sum()),
    ("delay_rate", round(model_df["is_delayed"].mean(), 4)),
    ("declared_value_mean_inr", round(model_df["declared_value_inr"].mean(), 4)),
    ("declared_value_median_inr", round(model_df["declared_value_inr"].median(), 4)),
    ("log_declared_value_mean", round(model_df["log_declared_value"].mean(), 4)),
    ("log_declared_value_median", round(model_df["log_declared_value"].median(), 4)),
    ("hours_q1", round(q1, 4)),
    ("hours_q3", round(q3, 4)),
    ("hours_lower_bound", round(lower_bound, 4)),
    ("hours_upper_bound", round(upper_bound, 4)),
    ("hours_outlier_count", outliers)
]

checks_df = pd.DataFrame(checks_data, columns=["metric", "value"])
checks_df.to_csv("outputs/part_a_checks.csv", index=False)


# -----------------------------------------------------------------
# TASK 6: Delay evidence table (outputs/part_a_delay_by_group.csv)
# -----------------------------------------------------------------
# Dimension 1: service_type
g1 = model_df.groupby("service_type", as_index=False).agg(
    shipments=("shipment_id", "count"),
    delayed=("is_delayed", "sum"),
    delay_rate=("is_delayed", "mean"),
    mean_hours=("delivery_hours", "mean")
)
g1.insert(0, "dimension", "service_type")
g1 = g1.rename(columns={"service_type": "group"})

# Dimension 2: hub_tier
g2 = model_df.groupby("hub_tier", as_index=False).agg(
    shipments=("shipment_id", "count"),
    delayed=("is_delayed", "sum"),
    delay_rate=("is_delayed", "mean"),
    mean_hours=("delivery_hours", "mean")
)
g2.insert(0, "dimension", "hub_tier")
g2 = g2.rename(columns={"hub_tier": "group"})
g2["group"] = g2["group"].astype(str)

# Dimension 3: distance_band (bins: 0-200, 201-400, 401-600, 601+)
model_df["distance_band"] = pd.cut(
    model_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True
)
g3 = model_df.groupby("distance_band", observed=False, as_index=False).agg(
    shipments=("shipment_id", "count"),
    delayed=("is_delayed", "sum"),
    delay_rate=("is_delayed", "mean"),
    mean_hours=("delivery_hours", "mean")
)
g3.insert(0, "dimension", "distance_band")
g3 = g3.rename(columns={"distance_band": "group"})
g3["group"] = g3["group"].astype(str)

# Combine and round to 4 decimal places
delay_table = pd.concat([g1, g2, g3], ignore_index=True)
delay_table["delay_rate"] = delay_table["delay_rate"].round(4)
delay_table["mean_hours"] = delay_table["mean_hours"].round(4)
delay_table.to_csv("outputs/part_a_delay_by_group.csv", index=False)


# ---------------------------------------------------------------------
# TASK 7: Correlations (outputs/part_a_correlations.csv)
# ---------------------------------------------------------------------
numeric_df = model_df.select_dtypes(include=[np.number])
corr_df = numeric_df.corr().round(4)
corr_df.insert(0, "column", corr_df.index)
corr_df.to_csv("outputs/part_a_correlations.csv", index=False)

print("Part A (Tasks 4, 5, 6, 7) completed successfully!")
