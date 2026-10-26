# =====================================================================
# PART D — PACKAGE THE MODEL (Task 5)
# =====================================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# Stratified split matching Part C
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Read chosen threshold
if os.path.exists("outputs/part_c_final_config.csv"):
    cfg_df = pd.read_csv("outputs/part_c_final_config.csv")
    chosen_threshold = float(cfg_df["threshold"].iloc[0])
else:
    chosen_threshold = 0.4

# Train final pipeline
final_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=1, random_state=42))
])
final_pipeline.fit(X_train, y_train)

# Save artifacts
joblib.dump(final_pipeline, "models/delay_model.pkl")

with open("models/delay_threshold.csv", "w") as f:
    f.write(f"{chosen_threshold}\n")

with open("models/feature_order.csv", "w") as f:
    f.write("feature\n")
    for feat in FEATURES:
        f.write(f"{feat}\n")

raw_shipments_df = pd.read_csv("data/raw_shipments.csv")
_, medians_dict = clean_shipments(raw_shipments_df)
impute_df = pd.DataFrame(list(medians_dict.items()), columns=["column", "median"])
impute_df.to_csv("models/impute_values.csv", index=False)

# Reload check
loaded_model = joblib.load("models/delay_model.pkl")
test_probs_orig = final_pipeline.predict_proba(X_test)[:, 1]
test_probs_loaded = loaded_model.predict_proba(X_test)[:, 1]
max_diff = float(np.abs(test_probs_orig - test_probs_loaded).max())

reload_check_df = pd.DataFrame([
    ("test_rows_scored", len(X_test)),
    ("reload_max_abs_probability_diff", max_diff)
], columns=["metric", "value"])
reload_check_df.to_csv("outputs/part_d_reload_check.csv", index=False)

print("Model packaged and reload check saved successfully!")
