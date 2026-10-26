# First import all the required libraries for the regression tasks.
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.shipment_prep import FEATURES, NOISE_COLUMNS

# Create required outputs and models directories if they do not exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------
# TASK 1: Stratified split into training, validation, and test sets
# ------------------------------------------------------------------
# Read the modelling table created in Part A
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# First split: Separate 70% for training and 30% temporary with stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Second split: Split the 30% temporary part equally into validation and test with stratification
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# Helper function to compute classification evaluation metrics
def get_metrics_row(model_name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": model_name,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4)
    }


# --------------------------------------------------------------------------
# TASK 2: Blind baseline and three classifiers evaluation on validation set
# --------------------------------------------------------------------------
models_rows = []

# 1. Blind baseline (predicts 0 / not delayed for every parcel)
blind_preds = np.zeros_like(y_val)
models_rows.append(get_metrics_row("blind_all_on_time", y_val, blind_preds))

# 2. Logistic Regression (inside Pipeline with StandardScaler)
lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
lr_pipe.fit(X_train, y_train)
models_rows.append(get_metrics_row("logistic_regression", y_val, lr_pipe.predict(X_val)))

# 3. Decision Tree Classifier (choosing best depth from 2 to 10 by highest validation F1)
best_dt_depth = 2
best_dt_f1 = -1
for d in range(2, 11):
    dt_temp = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    dt_temp.fit(X_train, y_train)
    f1_val = f1_score(y_val, dt_temp.predict(X_val), zero_division=0)
    if f1_val > best_dt_f1:
        best_dt_f1 = f1_val
        best_dt_depth = d

dt_final = DecisionTreeClassifier(criterion="gini", max_depth=best_dt_depth, random_state=42)
dt_final.fit(X_train, y_train)
models_rows.append(get_metrics_row(f"decision_tree_{best_dt_depth}", y_val, dt_final.predict(X_val)))

# 4. Random Forest Default Classifier
rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
rf_def.fit(X_train, y_train)
models_rows.append(get_metrics_row("random_forest_default", y_val, rf_def.predict(X_val)))
# Save model comparison metrics to outputs/part_c_models.csv
pd.DataFrame(models_rows).to_csv("outputs/part_c_models.csv", index=False)


# -----------------------------------------------
# TASK 3: Feature importance and noise selection
# -----------------------------------------------
# Fit a default forest on all 17 features to get feature importances
rf_feat = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feat.fit(X_train, y_train)
rf_importances = rf_feat.feature_importances_

# Fit SelectKBest with k="all" to get F-scores per column
selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)
f_scores = selector_all.scores_

feat_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances
})
feat_scores_df["f_rank"] = feat_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
feat_scores_df["is_noise"] = feat_scores_df["feature"].isin(NOISE_COLUMNS)
feat_scores_df = feat_scores_df.sort_values(by="f_rank")
feat_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

# Test selectk with k in [5, 7, 10, 12, 17]
selectk_rows = []
for k_val in [5, 7, 10, 12, 17]:
    pipe_k = Pipeline([
        ("selector", SelectKBest(f_classif, k=k_val)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe_k.fit(X_train, y_train)
    k_preds = pipe_k.predict(X_val)
    selectk_rows.append({
        "k": k_val,
        "val_f1": round(f1_score(y_val, k_preds, zero_division=0), 4)
    })
pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)


# ----------------------------------------------------------------
# TASK 4: Hyperparameter tuning (GridSearch and RandomizedSearch)
# ----------------------------------------------------------------
# Transform training and validation data using SelectKBest with k=7
selector_k7 = SelectKBest(f_classif, k=7)
X_train_sel = selector_k7.fit_transform(X_train, y_train)
X_val_sel = selector_k7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

rf_tune = RandomForestClassifier(random_state=42)

# GridSearchCV (27 candidates * 5 folds = 135 total fits)
grid_search = GridSearchCV(estimator=rf_tune, param_grid=param_grid, cv=5)
grid_search.fit(X_train_sel, y_train)

# RandomizedSearchCV (9 candidates * 5 folds = 45 total fits)
rand_search = RandomizedSearchCV(estimator=rf_tune, param_distributions=param_grid, n_iter=9, cv=5, random_state=42)
rand_search.fit(X_train_sel, y_train)

# Untuned forest on the same 7 selected features
untuned_rf = RandomForestClassifier(n_estimators=100, random_state=42)
untuned_rf.fit(X_train_sel, y_train)
untuned_preds = untuned_rf.predict(X_val_sel)
untuned_val_f1 = f1_score(y_val, untuned_preds, zero_division=0)

grid_preds = grid_search.predict(X_val_sel)
grid_val_f1 = f1_score(y_val, grid_preds, zero_division=0)

rand_preds = rand_search.predict(X_val_sel)
rand_val_f1 = f1_score(y_val, rand_preds, zero_division=0)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": round(grid_search.best_score_, 4),
        "val_f1": round(grid_val_f1, 4)
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(rand_search.best_params_),
        "best_cv_accuracy": round(rand_search.best_score_, 4),
        "val_f1": round(rand_val_f1, 4)
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": round(untuned_val_f1, 4)
    }
]
pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)
# -----------------------------------------
# TASK 5: Choosing the final configuration
# -----------------------------------------
tuning_df = pd.DataFrame(tuning_rows)
best_row = tuning_df.sort_values(by="val_f1", ascending=False).iloc[0]
chosen_search = best_row["search"]

if chosen_search == "grid_search":
    best_params = grid_search.best_params_
    chosen_from = "grid_search"
elif chosen_search == "randomized_search":
    best_params = rand_search.best_params_
    chosen_from = "randomized_search"
else:
    best_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    chosen_from = "default_forest_same_7_features"

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **best_params))
])
final_model.fit(X_train, y_train)

config_row = [{
    "selector_k": 7,
    "n_estimators": best_params.get("n_estimators", 100),
    "max_depth": best_params.get("max_depth", "unlimited"),
    "min_samples_leaf": best_params.get("min_samples_leaf", 1),
    "threshold": 0.5,
    "chosen_from": chosen_from
}]
pd.DataFrame(config_row).to_csv("outputs/part_c_final_config.csv", index=False)


# -----------------------------------------------------------
# TASK 6: Alert threshold selection based on operations rule
# -----------------------------------------------------------
val_probs = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresh_rows = []
valid_thresholds = []

for t in thresholds:
    t_preds = (val_probs >= t).astype(int)
    row = get_metrics_row(f"thresh_{t}", y_val, t_preds)
    row["threshold"] = t
    cols = ["threshold"] + [c for c in row.keys() if c != "threshold" and c != "model"]
    row_clean = {c: row[c] for c in cols}
    thresh_rows.append(row_clean)
    
    # Operations rule: recall must be at least 0.80
    if row["recall"] >= 0.80:
        valid_thresholds.append(t)

pd.DataFrame(thresh_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

# Choose highest threshold with recall >= 0.80 (fallback to 0.1 if none reaches)
if len(valid_thresholds) > 0:
    chosen_threshold = max(valid_thresholds)
else:
    chosen_threshold = 0.1

# Update final config with chosen threshold
config_row[0]["threshold"] = chosen_threshold
pd.DataFrame(config_row).to_csv("outputs/part_c_final_config.csv", index=False)


# ----------------------------------------------------------------
# TASK 7: One test evaluation using the final model and threshold
# ----------------------------------------------------------------
test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= chosen_threshold).astype(int)
test_row = get_metrics_row("final_test", y_test, test_preds)
test_row["threshold"] = chosen_threshold
cols_test = ["threshold"] + [c for c in test_row.keys() if c != "threshold" and c != "model"]
test_row_clean = {c: test_row[c] for c in cols_test}
pd.DataFrame([test_row_clean]).to_csv("outputs/part_c_final_test.csv", index=False)

print("Part C classification script completed successfully!")
print(f"Chosen Alert Threshold: {chosen_threshold}")
