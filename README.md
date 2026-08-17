# Telco Customer Churn

This project implements the workflow specified in `problem-statement.md` for the Kaggle Telco Customer Churn dataset. The only predictive model families are XGBoost and CatBoost.

## Dataset

Download source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

The CSV must be stored at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`. The supplied project copy was downloaded from Kaggle's public dataset endpoint. `TotalCharges` is parsed with coercion because the source contains blank numeric values. `customerID` is retained only for reporting and never used as a predictor.

## Setup and Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
pytest -q
```

`main.py` performs the initial audit, writes the cleaned interim data, creates a stratified 80/20 split, engineers leakage-safe features, trains the baseline XGBoost pipeline, and saves metrics, predictions, and the model artifact. The remaining modules provide the statistical, nonlinearity, selection, calibration, threshold, interpretation, and CatBoost building blocks required by the specification.

## Methodology

Development uses stratified cross-validation and PR-AUC as the tuning objective, with ROC-AUC, precision, recall, specificity, F1, log loss, and Brier score reported. Statistical tests include Mann-Whitney U for numeric comparisons, chi-square/Cramer's V for categorical associations, and Benjamini-Hochberg correction. Mutual information, binned churn rates, permutation importance, model gain, native CatBoost importance, and SHAP are interpretation or selection aids, not causal evidence.

XGBoost receives imputed and one-hot encoded categoricals. CatBoost receives categoricals as strings and uses native categorical handling. Thresholds are selected on development/validation data only; the final test set remains untouched until evaluation. Cost-based thresholds use explicit business assumptions.

## Graphify

Graphify is maintained as the repository knowledge graph. After meaningful changes, run:

```powershell
python -m graphify . --update --no-viz
```

or use the Graphify executable if installed. Generated `graphify-out/` artifacts are not model inputs.

## Reports and notebooks

The numbered notebooks mirror the 12 stages in the specification. They are intentionally thin execution documents that call reusable functions from `src/`, keeping analysis reproducible and avoiding duplicated notebook logic. Results must be generated from the downloaded data; no metrics are fabricated in committed documentation.
