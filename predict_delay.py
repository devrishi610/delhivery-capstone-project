# =====================================================================
# PART D — SCORING SCRIPT (Task 6)
# =====================================================================
import os
import numpy as np
import pandas as pd
import joblib

from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)

# Load artifacts
loaded_model = joblib.load("models/delay_model.pkl")
threshold = float(open("models/delay_threshold.csv").read().strip())
impute_df = pd.read_csv("models/impute_values.csv")
medians_dict = dict(zip(impute_df["column"], impute_df["median"]))

hub_segments = pd.read_csv("outputs/hub_segments.csv")
hub_seg_map = hub_segments.set_index("hub_id")["segment"].to_dict()

hubs_df = pd.read_csv("data/hubs.csv")
new_raw = pd.read_csv("data/new_shipments.csv") if os.path.exists("data/new_shipments.csv") else pd.read_csv("data/raw_shipments.csv").head(48)

# Clean and build features
new_clean, _ = clean_shipments(new_raw, medians=medians_dict)
new_feat = build_features(new_clean, hubs_df)
X_new = new_feat[FEATURES]

# Predict probabilities and apply operations rule actions
new_probs = loaded_model.predict_proba(X_new)[:, 1]
new_feat["delay_probability"] = np.round(new_probs, 4)
new_feat["flagged"] = (new_feat["delay_probability"] >= threshold).astype(int)
new_feat["segment"] = new_feat["hub_id"].map(hub_seg_map)

actions = []
for idx, row in new_feat.iterrows():
    flagged = row["flagged"]
    seg = row["segment"]
    if flagged == 1 and seg == "Escalate":
        action = "Escalate to hub manager"
    elif flagged == 1:
        action = "Proactive customer call"
    else:
        action = "No action"
    actions.append({
        "shipment_id": row["shipment_id"],
        "hub_id": row["hub_id"],
        "segment": seg,
        "delay_probability": row["delay_probability"],
        "flagged": flagged,
        "action": action
    })

actions_df = pd.DataFrame(actions)
actions_df.to_csv("outputs/new_shipment_actions.csv", index=False)

print("Scoring script executed successfully! new_shipment_actions.csv generated.")
