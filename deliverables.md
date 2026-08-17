# Customer Churn Model Deliverables

## Objective

Build a reproducible churn-prediction workflow using CatBoost and XGBoost. The project evaluates raw, engineered, and notebook-selected feature representations; tunes each allowed model per representation; selects an operating threshold on validation data; evaluates once on an untouched test split; and saves trained-model artifacts.

## Data and Assumptions

- Dataset: Kaggle Telco Customer Churn dataset, 7,043 customers.
- Target: `Churn`, encoded as `1` for churned and `0` for retained customers.
- Split strategy: stratified 64% training, 16% validation, and 20% untouched test data with `random_state=42`.
- `customerID` is retained only for prediction reporting and excluded from model inputs.
- Blank `TotalCharges` values are converted to missing values. CatBoost numeric missing values are filled from training-data medians only; categorical missing values are assigned the training-safe `Missing` category.
- Threshold selection minimizes an assumed business cost where a false negative costs twice as much as a false positive. These costs are assumptions and should be replaced with business-provided values.

## Feature Representations

Three feature representations were evaluated with both CatBoost and XGBoost.

| Representation | Description                                                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Raw            | Original customer attributes, with CatBoost handling categorical fields natively.                                                       |
| Engineered     | Raw fields plus controlled tenure, billing, service, profile, contract, and digital-engagement features from `src.feature_engineering`. |
| Selected       | Evidence-supported subset based on feature engineering, statistical/MI evidence, and prior feature-selection experiments.               |

## Hyperparameter Tuning

Each CatBoost representation used a bounded native validation search with early stopping. Each XGBoost representation used a bounded validation search after training-only imputation and one-hot encoding. The candidate settings varied:

- `iterations`
- `depth`
- `learning_rate`
- `l2_leaf_reg`
- `random_strength`
- `bagging_temperature`
- XGBoost tree count, depth, learning rate, row/column sampling, and minimum child weight

Candidates were ranked on validation PR-AUC. The test set was not used for tuning or threshold choice.

## Results

| Representation | Model    | Validation PR-AUC | Validation F1 | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 | Threshold |
| -------------- | -------- | ----------------: | ------------: | -----------: | ----------: | -------------: | ----------: | ------: | --------: |
| Raw            | CatBoost |            0.6760 |        0.6527 |       0.8443 |      0.6671 |         0.5885 |      0.6578 |  0.6212 |      0.40 |
| Engineered     | CatBoost |            0.6751 |        0.6514 |       0.8463 |      0.6639 |         0.5727 |      0.7059 |  0.6323 |      0.35 |
| Selected       | CatBoost |            0.6752 |        0.6488 |       0.8455 |      0.6584 |         0.5947 |      0.6551 |  0.6234 |      0.40 |
| Raw            | XGBoost  |            0.6747 |        0.6525 |       0.8451 |      0.6611 |         0.5976 |      0.6551 |  0.6250 |      0.40 |
| Engineered     | XGBoost  |            0.6736 |        0.6525 |       0.8419 |      0.6526 |         0.5714 |      0.6310 |  0.5997 |      0.40 |
| Selected       | XGBoost  |            0.6728 |        0.6606 |       0.8460 |      0.6579 |         0.5681 |      0.7139 |  0.6327 |      0.35 |

## Model Selection

The best model is the **raw-feature CatBoost model** because it has the highest validation PR-AUC (`0.6760`) and the highest held-out test PR-AUC (`0.6671`). CatBoost uses categorical fields directly, avoiding one-hot encoding for the winning representation.

The selected-feature XGBoost model has the strongest validation F1 (`0.6606`) and test recall (`0.7139`), making it a useful recall-oriented retention-campaign alternative. It does not exceed raw CatBoost on PR-AUC, so it is not the primary ranking model.

## Saved Artifacts

The winning model and its artifacts are saved in:

- `models/best_model/model.cbm`: native CatBoost model for the winning experiment
- `models/best_model/model.joblib`: Python-serialized winning model
- `models/best_model/metrics.json`: test and validation metrics
- `models/best_model/metadata.json`: feature columns, categorical columns, and tuned parameters
- `models/best_model/roc_auc.png`: held-out test ROC-AUC curve
- `models/best_model.json`: winning-experiment manifest
- `outputs/tables/experiment_comparison.csv`: feature-set comparison
- `outputs/predictions/raw_catboost_test.csv`: customer-level held-out test predictions
- `outputs/figures/*_catboost_roc_auc.png` and `outputs/figures/*_xgboost_roc_auc.png`: ROC-AUC curves for every experiment

## 90-95% Precision, Recall, and F1 Target

The requested simultaneous 90-95% precision, recall, and F1 was not achieved and is not a credible target for this dataset under a leakage-safe evaluation. The winning ranking model has held-out F1 `0.6212`, precision `0.5885`, and recall `0.6578`; the recall-oriented selected XGBoost alternative has test F1 `0.6327`, precision `0.5681`, and recall `0.7139`.

Precision and recall trade off as the decision threshold moves. Raising recall typically increases false positives and lowers precision; raising precision typically misses more churners. Reporting 90-95% for all three metrics would require substantially more predictive signal, changed label definition, leakage, or a non-comparable split. This project preserves the more defensible result rather than optimizing to an unsupported target.

## Conclusions

- CatBoost is the strongest ranking model here because it accepts categorical variables directly and reduces the need for one-hot encoding.
- XGBoost is a useful complementary comparison. Its selected-feature representation offers the highest recall after validation threshold selection.
- Contract, service, charge, tenure, and payment-related variables provide meaningful churn information, but raw CatBoost already captures that information effectively.
- Validation-selected thresholds of `0.35` to `0.40` prioritize recall due to the assumed higher cost of missing a likely churner.
- Model explanations and retention decisions must be interpreted as predictive associations, not causal claims.

## Next Steps

1. Replace assumed false-negative and false-positive costs with business-provided campaign economics.
2. Add customer value or margin so retention prioritization considers churn probability multiplied by value.
3. Collect temporal behavior, service-usage, complaint, interaction, and campaign-response data; the current dataset has limited behavioral context.
4. Evaluate calibration and optionally calibrate only through training/validation workflows.
5. Monitor data drift, performance drift, segment fairness, and threshold stability after deployment.
6. Run controlled retention experiments before treating any model-identified pattern as an intervention recommendation.

## Other Model Families

The project specification allows only XGBoost and CatBoost, so no other predictive model was added to the experiment. If that restriction changes, suitable comparison candidates are LightGBM for categorical/tree boosting efficiency, regularized logistic regression for a transparent linear benchmark, and calibrated histogram gradient boosting for a compact tree baseline. They should be added only after the project’s strict-modeling requirement is formally relaxed.
