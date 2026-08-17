# Telco Customer Churn Report

This report is generated from the workflow in `problem-statement.md`. Numeric results are intentionally populated only after running the pipeline against the Kaggle CSV.

## Executive Summary

Run `python main.py` and the numbered notebooks to generate evidence-based conclusions. No result in this report should be treated as final until it is backed by an output artifact.

## 1. Business Problem

Predict churn risk and translate model behavior into retention prioritization. The objective is not accuracy alone.

## 2. Dataset

Kaggle Telco Customer Churn. `customerID` is an identifier, `Churn` is the binary target, and blank `TotalCharges` values are safely coerced to missing.

## 3. Data Quality

See `outputs/tables/data_audit.json` after execution.

## 4. Exploratory Data Analysis

Use notebook 02 for distributions, counts, churn rates, and numerical/categorical summaries.

## 5. Statistical Testing

Use numerical effect sizes, categorical Cramer's V, and Benjamini-Hochberg FDR correction. Association is not causation.

## 6. Linearity vs Nonlinearity

Use quantile churn-rate bins, log-odds diagnostics, transformations, MI, and model explanations. Normality is not a linearity test.

## 7. Feature Engineering

Features cover tenure, billing consistency, service counts, customer profile, contracts, and digital engagement. Ratios protect zero-tenure rows.

## 8. Feature Selection

Compare filter, XGBoost embedded, CatBoost embedded, permutation, SHAP, and cross-fold stability evidence without inspecting the test set.

## 9-13. Models and Evaluation

Compare baseline XGBoost, tuned XGBoost, and tuned CatBoost with stratified cross-validation and test-set metrics.

## 14. Statistical Model Comparison

Use paired procedures only after predictions are generated on the same untouched test observations.

## 15. Threshold Optimization

Choose a threshold using development data and explicit FN/FP assumptions, then lock it before final test evaluation.

## 16. Probability Calibration

Report calibration curves, Brier score, and log loss. Calibration is not automatic.

## 17. Model Interpretation

SHAP and feature importance explain model behavior, not causal drivers.

## 18. High-Risk Customer Analysis

Use scored customer output to profile high-risk segments by contract, tenure, charges, services, payment method, and support.

## 19. Business Recommendations

Prioritize experiments around month-to-month contracts, early-tenure onboarding, high charges, support friction, and payment-method friction only when observed evidence supports them.

## 20. Model Limitations

Consider sample representativeness, historical bias, missing behavior, temporal drift, imbalance, calibration, stability, fairness, and cost assumptions.

## 21. Future Improvements

Add temporal customer history, campaign outcomes, customer value, drift monitoring, and controlled retention experiments.

## 22. Final Conclusion

Select the final model using predictive performance, recall, precision, PR-AUC, calibration, stability, interpretability, business cost, and operational complexity together.
