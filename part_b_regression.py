# First import all the required libraries for the regression tasks.
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

# Create required outputs and modelsdiectories if they do not already exist. 
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------------------
# TASK 1: Split into training, validation, and test sets (840 / 180 / 180)
# -------------------------------------------------------------------------
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]

# First split : 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Second split: 50% validation, 50% test from temporary
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# ------------------------------------------------------------
# TASK 2: Scaling without leakage (fit on training rows only)
# ------------------------------------------------------------
# Initialize StandardScaler and fit exclusively on the training rows to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Calculate maximum absolute mean to verify proper scaling
train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
test_scaled_max_abs_mean = float(np.abs(X_val_scaled.mean(axis=0)).max())

# Save scaling and split size checks to outputs/part_b_checks.csv
checks_data = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
    ("test_scaled_max_abs_mean", test_scaled_max_abs_mean)
]
pd.DataFrame(checks_data, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)


# Helper function to compute evaluation metrics (MAE, MSE, RMSE, R2 score)
def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return round(mae, 4), round(mse, 4), round(rmse, 4), round(r2, 4)


# --------------------------------------------------------------------------
# TASK 3: Baseline model (predict mean of y_train for every validation row)
# --------------------------------------------------------------------------
# Create baseline predictions using the mean of the training target
val_baseline_preds = np.full_like(y_val, y_train.mean())
base_mae, base_mse, base_rmse, base_r2 = evaluate_metrics(y_val, val_baseline_preds)


# ----------------------------------------------------------
# TASK 4: Tree-depth diagnostic for Decision Tree Regressor
# ----------------------------------------------------------
# Test multiple maximum depths to evaluate train vs validation performance
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
best_val_rmse = float('inf')
best_depth_val = 4  # default fallback depth

for d in depths:
    dt_temp = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt_temp.fit(X_train, y_train)
    
    tr_rmse = np.sqrt(mean_squared_error(y_train, dt_temp.predict(X_train)))
    v_rmse = np.sqrt(mean_squared_error(y_val, dt_temp.predict(X_val)))
    
    # Track the depth that yields the lowest validation RMSE
    if v_rmse < best_val_rmse:
        best_val_rmse = v_rmse
        best_depth_val = d
        
    depth_rows.append({
        "max_depth": str(d) if d is not None else "unlimited",
        "train_rmse": round(tr_rmse, 4),
        "val_rmse": round(v_rmse, 4)
    })

# Save tree depth diagnostics to outputs/part_b_tree_depth.csv 
pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)


# -----------------------------------------------------------
# TASK 5: Four candidate models evaluation on validation set
# -----------------------------------------------------------
# 1. Linear Regression (wrapped inside a Pipeline with StandardScaler)
lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
lr_pipe.fit(X_train, y_train)
lr_mae, lr_mse, lr_rmse, lr_r2 = evaluate_metrics(y_val, lr_pipe.predict(X_val))

# 2. Decision Tree Regressor (using the best depth found in task 4)
dt_final = DecisionTreeRegressor(max_depth=best_depth_val, random_state=42)
dt_final.fit(X_train, y_train)
dt_mae, dt_mse, dt_rmse, dt_r2 = evaluate_metrics(y_val, dt_final.predict(X_val))

# 3. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_metrics(y_val, rf.predict(X_val))

# 4. Gradient Boosting Regressor
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_mae, gb_mse, gb_rmse, gb_r2 = evaluate_metrics(y_val, gb.predict(X_val))

# Compile validation metrics for all models
metrics_rows = [
    {"model": "mean_baseline", "split": "val", "MAE": base_mae, "MSE": base_mse, "RMSE": base_rmse, "R2": base_r2},
    {"model": "linear_regression", "split": "val", "MAE": lr_mae, "MSE": lr_mse, "RMSE": lr_rmse, "R2": lr_r2},
    {"model": "decision_tree", "split": "val", "MAE": dt_mae, "MSE": dt_mse, "RMSE": dt_rmse, "R2": dt_r2},
    {"model": "random_forest", "split": "val", "MAE": rf_mae, "MSE": rf_mse, "RMSE": rf_rmse, "R2": rf_r2},
    {"model": "gradient_boosting", "split": "val", "MAE": gb_mae, "MSE": gb_mse, "RMSE": gb_rmse, "R2": gb_r2},
]


# -------------------------------------------------------
# TASK 6: Learning-rate experiment for Gradient Boosting
# -------------------------------------------------------
# Test different learning rates (0.01, 0.1, 1.0) and record validation performance
lr_exp_rows = []
for lr_val in [0.01, 0.1, 1.0]:
    gb_exp = GradientBoostingRegressor(n_estimators=100, learning_rate=lr_val, random_state=42)
    gb_exp.fit(X_train, y_train)
    preds = gb_exp.predict(X_val)
    lr_exp_rows.append({
        "learning_rate": lr_val,
        "val_rmse": round(np.sqrt(mean_squared_error(y_val, preds)), 4),
        "val_r2": round(r2_score(y_val, preds), 4)
    })
pd.DataFrame(lr_exp_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)


# -------------------------------------------------------------
# TASK 7: Cross-validated comparison and final model selection
# -------------------------------------------------------------
# Rank candidate models by validation RMSE to identify the top 2 models
candidates = {
    "linear_regression": lr_rmse,
    "gradient_boosting": gb_rmse,
    "random_forest": rf_rmse,
    "decision_tree": dt_rmse
}
sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])
top1_model_name = sorted_candidates[0][0]
top2_model_name = sorted_candidates[1][0]

def get_model_object(name):
    if name == "linear_regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    elif name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        return DecisionTreeRegressor(max_depth=best_depth_val, random_state=42)

# Perform 5-fold cross-validation on combined training and validation data (1020 rows)
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for m_name in [top1_model_name, top2_model_name]:
    model_obj = get_model_object(m_name)
    scores = cross_val_score(model_obj, X_train_val, y_train_val, cv=kf, scoring="r2")
    cv_rows.append({
        "model": m_name,
        "mean_r2": round(float(scores.mean()), 4),
        "std_r2": round(float(scores.std()), 4)
    })
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

# Train final winning model on training rows and evaluate exactly once on test rows
final_model = get_model_object(top1_model_name)
final_model.fit(X_train, y_train)
test_mae, test_mse, test_rmse, test_r2 = evaluate_metrics(y_test, final_model.predict(X_test))

# Append final test row to metrics CSV
metrics_rows.append({
    "model": top1_model_name,
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
})

pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)

print("Part B execution completed successfully!")
print(f"Winning Model: {top1_model_name} with Test R2: {test_r2}")
