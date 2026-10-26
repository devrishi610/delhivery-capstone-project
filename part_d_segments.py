# =====================================================================
# PART D — HUB SEGMENTS & OPERATIONS PLAYBOOK (Tasks 1 to 4)
# =====================================================================
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

os.makedirs("outputs", exist_ok=True)

# Read modelling table and hubs data
df = pd.read_csv("data/modelling_table.csv")
hubs_df = pd.read_csv("data/hubs.csv")

# ---------------------------------------------------------------------
# TASK 1: Hub profile aggregation (outputs/hub_profile.csv)
# ---------------------------------------------------------------------
hub_profile = df.groupby("hub_id", as_index=False).agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).sort_values("hub_id").reset_index(drop=True)

hub_profile = hub_profile[["hub_id", "shipments", "mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]]
hub_profile.to_csv("outputs/hub_profile.csv", index=False)


# ---------------------------------------------------------------------
# TASK 2 & 3: K-Means Segmentation & Separation Evidence
# ---------------------------------------------------------------------
X_hubs = hub_profile[["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]]
scaler_hub = StandardScaler()
X_hubs_scaled = scaler_hub.fit_transform(X_hubs)

# Separation evidence for K in 2, 3, 4, 5
sep_rows = []
for k_val in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    labels = km.fit_predict(X_hubs_scaled)
    
    dist_matrix = euclidean_distances(X_hubs_scaled)
    same_cluster_dists, diff_cluster_dists = [], []
    n_hubs = len(labels)
    
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            d = dist_matrix[i, j]
            if labels[i] == labels[j]:
                same_cluster_dists.append(d)
            else:
                diff_cluster_dists.append(d)
                
    mean_within = np.mean(same_cluster_dists) if same_cluster_dists else 1.0
    mean_across = np.mean(diff_cluster_dists) if diff_cluster_dists else 1.0
    ratio = mean_across / mean_within
    
    cluster_sizes = sorted([np.sum(labels == c) for c in range(k_val)], reverse=True)
    cluster_sizes_str = "-".join(map(str, cluster_sizes))
    
    sep_rows.append({
        "K": k_val,
        "across_over_within": round(ratio, 4),
        "cluster_sizes": cluster_sizes_str
    })

pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)


# ---------------------------------------------------------------------
# TASK 4: Segment names and tier check
# ---------------------------------------------------------------------
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
hub_profile["cluster"] = kmeans_3.fit_predict(X_hubs_scaled)

# Order clusters by mean delay rate: Standard (lowest), Watch (middle), Escalate (highest)
cluster_delay = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values()
cluster_names_map = {}
names_list = ["Standard", "Watch", "Escalate"]
for idx, (c_id, mean_del) in enumerate(cluster_delay.items()):
    cluster_names_map[c_id] = names_list[idx]

hub_profile["segment"] = hub_profile["cluster"].map(cluster_names_map)

hub_merged = hub_profile.merge(hubs_df, on="hub_id", how="left")
hub_segments = hub_merged[["hub_id", "segment", "hub_city", "hub_tier"]]
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

segment_profile = hub_merged.groupby("segment", as_index=False).agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
)
segment_profile.to_csv("outputs/segment_profile.csv", index=False)

# Tier vs Segment mismatch
tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
hub_merged["implied_segment"] = hub_merged["hub_tier"].map(tier_map)
mismatches = hub_merged[hub_merged["segment"] != hub_merged["implied_segment"]]
mismatches[["hub_id", "hub_city", "hub_tier", "segment"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print("Part D Hub Segments successfully created!")
