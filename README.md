# Delhivery Last-Mile Delay Intelligence Pipeline

## 1. How to Run

This project analyses shipment data, predicts delivery delays, groups delivery hubs into different segments, and suggests actions for new shipments.

### Requirements

- Python
- pandas
- numpy
- scikit-learn
- joblib

### Steps to Run

Clone the repository and open the project folder. Run the following commands in order:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

These commands generate the data, perform data cleaning and analysis, train and evaluate the models, create hub segments, package the final model, and predict delays for new shipments.

Running `python package_model.py` regenerates `models/delay_model.pkl` and the other model files.

The committed model file is expected to load reliably with the scikit-learn version specified in `requirements.txt`. If the installed versions differ, the model should be regenerated using `package_model.py`.

## 2. Data Findings

The project uses shipment data containing information about delivery time, distance, service type, hub details, and other shipment-related features.

The raw dataset contains 1,224 rows. After removing duplicate records, 1,200 rows remain for further analysis.

Some important findings from the analysis are:

- Express shipments have a higher delay rate than Standard and Economy shipments.
- Delay rates differ across hub tiers.
- Shipments covering longer distances generally show a higher delay rate.
- Distance, promised delivery time, and hub-related features help explain differences in delivery delays.
- The declared value of shipments also varies across the dataset.

The following figures should be checked against the generated output files before submission:

| Finding | Result |
|---|---:|
| Raw shipment records | 1,224 |
| Records after removing duplicates | 1,200 |
| Express delay rate | 48.97% |
| Standard delay rate | 20.56% |
| Economy delay rate | 8.57% |
| Tier 1 delay rate | 4.00% |
| Tier 2 delay rate | 21.50% |
| Tier 3 delay rate | 44.00% |

The analysis also compares delay rates across distance groups and examines the relationship between numerical features.

## 3. Model Results (Regression)

Regression models were used to estimate delivery time in hours.

A baseline model was compared with Linear Regression, Decision Tree, Random Forest, and Gradient Boosting. The models were evaluated using RMSE and R².

The baseline validation R² is approximately -0.021. A negative R² indicates that the baseline does not explain the variation in delivery time well.

The reference validation R² values are:

| Model | Validation R² |
|---|---:|
| Linear Regression | 0.9313 |
| Decision Tree | 0.8867 |
| Random Forest | 0.9128 |
| Gradient Boosting | 0.9306 |

The final model should be selected according to the lowest validation RMSE, as required by the project.

- **Best model:** Linear Regression
- **Best validation RMSE:** 4.8912
- **Best tree depth:** 6
- **Best learning rate:** 0.1

The tree-depth experiment helps compare high-bias and high-variance behaviour. The learning-rate experiment shows how different learning rates affect Gradient Boosting performance.

The cross-validation results are also considered when deciding which model is suitable for the project.

## 4. Model Results (Classification)

Classification was used to predict whether a shipment would be delayed.

Different classification models were compared using metrics such as accuracy, precision, recall, and F1-score. These metrics help evaluate the model beyond accuracy alone.

The baseline model has a validation accuracy of approximately 76.67%, but it does not identify delayed shipments effectively. Therefore, accuracy alone is not sufficient for this problem.

The analysis also examines feature importance. The top seven features in the reference results are:

1. `distance_km`
2. `hub_tier`
3. `hub_daily_capacity`
4. `n_stops`
5. `promised_hours`
6. `rain_flag`
7. `sale_event_flag`

The classification threshold is selected using the validation results and the project's recall requirement.

Final results should be checked against these files:

- `outputs/part_c_final_config.csv`
- `outputs/part_c_final_test.csv`
- `outputs/part_c_thresholds.csv`

**Chosen threshold:** 0.5

**Test Accuracy:** 0.8667

**Test Precision:** 0.6889

**Test Recall:** 0.7561

**Test F1-score:** 0.7209

Precision indicates how many flagged shipments are actually delayed, while recall indicates how many delayed shipments the model successfully identifies.

## 5. Operations Recommendation

The 24 delivery hubs are grouped into three segments using K-Means clustering:

- **Standard:** Hubs with the lowest average delay rate.
- **Watch:** Hubs with a middle-level average delay rate.
- **Escalate:** Hubs with the highest average delay rate.

The reference segment profiles are:

| Segment | Number of Hubs | Average Delay Rate |
|---|---:|---:|
| Standard | 7 | 0.025714285714285714 |
| Watch | 8 | 0.155 |
| Escalate | 9 | 0.4600000000000001 |

These figures should be verified against `outputs/segment_profile.csv`.

The hubs in the Escalate segment should receive priority for operational review. Their names and delay rates should be taken from `outputs/hub_segments.csv` and `outputs/segment_profile.csv`.

The project also identifies hubs whose assigned tier does not match their delay-based segment. The reference mismatches are:

- HUB07, Kolkata: Tier 1 but belongs to Watch.
- HUB13, Nagpur: Tier 2 but belongs to Escalate.
- HUB15, Kochi: Tier 2 but belongs to Escalate.
- HUB22, Dehradun: Tier 3 but belongs to Watch.

These mismatches show that a hub's tier label should not be used alone to decide its operational priority. Its observed delay performance should also be considered.

### Shipment Alert and Action Rules

The alert threshold is taken from the final classification configuration. A shipment is flagged when its delay probability is greater than or equal to this threshold.

The action depends on the shipment's predicted risk and hub segment:

- A flagged shipment belonging to an Escalate hub is escalated to the hub manager.
- A flagged shipment belonging to another segment receives a proactive customer call.
- A shipment that is not flagged requires no action.

The following results must be filled in from the generated output files:

- **Alert threshold:** 0.5
- **Precision:** 0.6889
- **Recall:** 0.7561
- **Total new shipments:** 48
- **Flagged shipments:** 6
- **Shipments requiring hub-manager escalation:** 5
- **Shipments requiring proactive customer call:** 1
- **Shipments requiring no action:** 42

K = 3 is retained because the operations team requires three segments with separate action plans. Although the separation ratio increases for K = 4 and K = 5, these solutions contain clusters with fewer than six hubs. Such small groups may be too small to justify a separate operational playbook. The comparison is available in `outputs/part_d_separation.csv`.

## 6. Production Plan

The project can be improved further before being used in a live delivery operation.

The following points describe how the current project supports production readiness:

- **Accuracy:** The regression and classification models are evaluated using separate metrics.
- **Reliability:** The model and its supporting files are saved for reuse.
- **Reproducibility:** The project provides scripts to regenerate the data, results, and model files.
- **Maintainability:** Data preparation and feature creation are shared through `src/shipment_prep.py`.
- **Consistency:** The same preprocessing functions and stored imputation values are used during training and prediction.
- **Performance:** The saved model can score new shipments without repeating the complete model-training process.
- **Monitorability:** The generated predictions and action file can be reviewed to understand which shipments have been flagged.

### Monitoring Plan

The following are proposed starting thresholds for monitoring. They should be reviewed and adjusted using actual operational data.

| Quantity to Monitor | Suggested Trigger | Action |
|---|---|---|
| Weekly delay rate | Increase of 5 percentage points compared with the relevant baseline | Investigate possible delivery problems |
| Share of shipments flagged | Change of 10 percentage points compared with the relevant baseline | Review shipment conditions and model behaviour |
| Recall on shipments with known outcomes | Falls below 0.80 | Review the threshold and model performance |

The model should be reviewed monthly, with retraining considered when performance deteriorates or sufficient new labelled data becomes available.

The sale window from 15 to 28 September should be compared with similar sale periods rather than ordinary weeks. This is important because `sale_event_flag` may affect shipment patterns and delay rates.

Data drift may appear as changes in shipment distance, promised delivery time, hub capacity, service type, or the frequency of sale events. These changes should be monitored because they may affect model performance.

## 7. Limitations

- The project uses synthetic shipment data, so the results may not represent actual delivery operations.
- The test dataset contains a limited number of delayed shipments, which can affect the reliability of evaluation metrics.
- The alert threshold is selected using one validation split, so its performance may vary on different data.
- The model depends on the available input features and may not account for every real-world reason for delivery delays.
- The project does not use a live shipment system or real-time operational data.

Therefore, the results should be treated as a project-based analysis and not as a guarantee of actual delivery times or delays.
