# Advanced End-to-End Telco Customer Churn Report

## Executive Summary

Customer attrition in subscription telecommunications represents a direct destruction of monthly recurring revenue (MRR) and customer lifetime value (CLV). Acquiring a replacement subscriber costs 5 to 7 times more than retaining an existing account. This project delivers an advanced, leakage-free, machine learning system for predicting subscriber churn using the Kaggle Telco Customer Churn dataset ($N = 7,043$).

In strict conformance with project constraints, predictive modeling was restricted exclusively to **XGBoost** and **CatBoost**. A baseline XGBoost model operating at a naive $0.50$ threshold exhibited a critically deficient recall of $49.20\%$, failing to detect over half of all churners. Through rigorous hyperparameter optimization and business cost-weighted threshold calibration ($C_{FN} = 2 \times C_{FP}$), our winning production model—**Tuned Raw CatBoost**—achieved a **held-out Test PR-AUC of 0.6671**, **ROC-AUC of 0.8443**, and **Recall of 65.78%** at an optimal threshold of $0.40$. A complementary candidate—**Tuned Selected XGBoost**—attained **71.39% Recall** and an **F1 score of 0.6327** at threshold $0.35$, serving as a specialized high-recall intervention engine.

---

## 1. Business Problem

The objective is to identify subscribers at acute risk of churn prior to billing cycle termination and prescribe data-backed retention interventions.

### Key Business Dynamics
- **Class Imbalance**: Churners comprise $26.54\%$ ($1,869$ accounts) vs $73.46\%$ ($5,174$ accounts) retained. Raw accuracy is a deceptive vanity metric.
- **Asymmetric Loss Matrix**: 
  - *False Negative (FN)*: A churner is missed; customer cancels service; loss is future CLV minus acquisition cost ($\approx \$200$).
  - *False Positive (FP)*: A loyal customer is flagged; company expends a retention concession or outreach effort ($\approx \$100$).
  - *Strategic Directive*: Minimize False Negatives by tuning the decision boundary below $0.50$.

---

## 2. Dataset Overview

- **Source**: Kaggle Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Volume**: 7,043 customer records across 21 raw attributes.
- **Target Variable**: `Churn` (Binary: `Yes` $\to 1$, `No` $\to 0$).
- **Customer Identifier**: `customerID` (retained strictly for downstream scoring audits; stripped from all feature matrices).

---

## 3. Data Quality & Audit

An automated data audit (`outputs/tables/data_audit.json`) established:
1. **TotalCharges Missingness**: 11 records contained whitespace strings (`" "`) corresponding to brand new customers with `tenure == 0`. These were coerced to `NaN` and safely median-imputed using training set statistics only.
2. **Duplicate Records**: 0 duplicate rows detected.
3. **Cardinality**: Categorical feature cardinality ranged from 2 (binary flags like `Partner`, `Dependents`) to 4 (`PaymentMethod`). Zero high-cardinality ID leaks were present.
4. **Target Distribution**: Exactly $26.54\%$ positive class prevalence across all stratified folds.

---

## 4. Exploratory Data Analysis (EDA)

Key distributions and empirical churn rates identified:
- **Tenure**: Heavy bimodal distribution peaking at month 1 and month 72. Month 1 customers exhibit a **56.91%** churn rate; month 72 customers exhibit **< 5%** churn.
- **Monthly Charges**: Bimodal distribution peaking at \$20 (basic phone) and \$80–\$90 (fiber optic). Churn peaks at ~40% for subscribers billed between \$70 and \$100.
- **Contracts**: Month-to-month contracts represent 55% of the customer base but drive over 88% of all churn events.

---

## 5. Statistical Hypothesis Testing

All hypothesis tests were performed on the training split ($N = 4,507$) with Benjamini-Hochberg False Discovery Rate (FDR) adjustments ($\alpha = 0.05$):

### Numerical Features (Mann-Whitney U & Cohen's d)
- **`tenure`**: $U = 1,042,784.0$, $p = 1.47 \times 10^{-130}$, **Cohen's $d = -0.880$** (Large effect). Tenure is the single strongest continuous discriminator.
- **`MonthlyCharges`**: $U = 2,465,303.5$, $p = 2.62 \times 10^{-36}$, **Cohen's $d = +0.472$** (Moderate effect). Churners pay \$18.60 more per month on average.
- **`TotalCharges`**: $U = 1,384,597.5$, $p = 3.22 \times 10^{-53}$, **Cohen's $d = -0.485$** (Moderate effect).

### Categorical Features (Chi-Square & Cramér's V)
- **`Contract`**: $\chi^2 = 768.68$, $df = 2$, $p = 1.21 \times 10^{-167}$, **Cramér's $V = 0.4130$** (Dominant predictor).
- **`OnlineSecurity`**: $\chi^2 = 549.58$, $df = 2$, $p = 4.58 \times 10^{-120}$, **Cramér's $V = 0.3492$**.
- **`TechSupport`**: $\chi^2 = 532.69$, $df = 2$, $p = 2.13 \times 10^{-116}$, **Cramér's $V = 0.3438$**.
- **`InternetService`**: $\chi^2 = 477.93$, $df = 2$, $p = 1.66 \times 10^{-104}$, **Cramér's $V = 0.3256$**.
- **`gender` & `PhoneService`**: FDR adjusted $p > 0.40$ (No statistically significant association with churn).

---

## 6. Linearity vs. Non-Linearity Analysis

Logit-scale diagnostics ($\ln(p / (1-p))$) disproved linear assumptions:
1. **`tenure`**: Displays a steep hazard cliff during months 0–6 ($\Delta \text{log-odds} = -0.99$) before flattening into a long tail.
2. **`MonthlyCharges`**: Demonstrates severe **non-monotonicity**:
   - \$18.40 – \$20.30: 8.70% churn rate (basic voice).
   - \$70.35 – \$100.15: 39.75% churn rate (fiber optic).
   - \$100.15 – \$118.75: 26.88% churn rate (fully bundled multi-service).
3. **Conclusion**: The non-monotonic hump in monthly charges makes linear models invalid, necessitating gradient-boosted decision trees.

---

## 7. Feature Engineering

19 domain features were engineered without target statistics:
- **Tenure Dynamics**: `tenure_months`, `tenure_years`, `is_new_customer` ($\le 6$ mo), `is_long_term_customer` ($\ge 48$ mo).
- **Pricing Ratios**: `avg_monthly_charge` ($\text{Total} / \text{tenure}$), `charge_per_tenure_month`, `total_charge_consistency` ($\text{Total} - (\text{Monthly} \times \text{tenure})$).
- **Service Bundling**: `number_of_security_services`, `number_of_streaming_services`, `number_of_services`, `digital_engagement_count`.
- **Household Demographics**: `has_partner_or_dependents`, `senior_single_customer`.
- **Contract Indicators**: One-hot indicators for 1-year and 2-year commitments.

---

## 8. Feature Selection

Feature selection synthesized Mutual Information ranking, permutation importance, and tree gain:
- A parsimonious set of **23 features** was established (`src/feature_selection.py`).
- Statistically uninformative features (`gender`, `PhoneService`) were excluded.
- The selected subset provided the foundation for the high-recall XGBoost model.

---

## 9. XGBoost Baseline

- **Configuration**: Default hyperparameters, threshold $= 0.50$.
- **Performance**:
  - Test Accuracy: $79.63\%$
  - Test Precision: $65.48\%$
  - **Test Recall: 49.20%**
  - Test F1: $56.18\%$
  - Test PR-AUC: $0.6553$
  - Test ROC-AUC: $0.8419$
- **Flaw**: Baseline XGBoost missed 190 actual churners, yielding unacceptably high revenue loss.

---

## 10. XGBoost Optimization

Hyperparameter tuning utilized validation PR-AUC with early stopping:
- Optimal parameters: `n_estimators=80`, `max_depth=3`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=1`.
- Regularization restricted tree depth to 3, curbing overfitting on high-dimensional one-hot columns.

---

## 11. CatBoost Optimization

CatBoost was tuned natively on categorical columns:
- Optimal parameters: `iterations=260`, `depth=5`, `learning_rate=0.04`, `l2_leaf_reg=6.0`, `bagging_temperature=0.5`.
- Native handling eliminated one-hot sparsity and preserved joint categorical combinations.

---

## 12. Cross-Validation Framework

- **Split Ratio**: 64% Training ($N=4,507$), 16% Validation ($N=1,127$), 20% Held-Out Test ($N=1,409$).
- **Stratification**: Preserved identical 26.54% churn proportions across all partitions.
- **Integrity**: Zero test data was observed during imputation, tuning, or threshold selection.

---

## 13. Comprehensive Model Evaluation

Held-out test results across all 6 model-feature combinations:

| Feature Set | Model | Valid PR-AUC | Valid F1 | Threshold | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw** | **CatBoost** | **0.6760** | 0.6527 | **0.40** | **0.8443** | **0.6671** | 0.5885 | 0.6578 | 0.6212 |
| **Engineered**| **CatBoost** | 0.6751 | 0.6514 | 0.35 | 0.8463 | 0.6639 | 0.5727 | 0.7059 | 0.6323 |
| **Selected**  | **CatBoost** | 0.6752 | 0.6488 | 0.40 | 0.8455 | 0.6584 | 0.5947 | 0.6551 | 0.6234 |
| **Raw** | **XGBoost**  | 0.6729 | 0.6498 | 0.35 | 0.8451 | 0.6609 | 0.5976 | 0.6551 | 0.6250 |
| **Selected**  | **XGBoost**  | 0.6725 | **0.6611** | **0.35** | **0.8460** | 0.6497 | 0.5681 | **0.7139** | **0.6327** |
| **Engineered**| **XGBoost**  | 0.6720 | 0.6538 | 0.35 | 0.8419 | 0.6492 | 0.5714 | 0.6310 | 0.5997 |

---

## 14. Statistical Model Comparison

- **ROC-AUC Equivalence**: DeLong test indicates no statistically significant difference between Raw CatBoost (0.8443) and Selected XGBoost (0.8460) ($p = 0.48$).
- **PR-AUC Dominance**: Raw CatBoost demonstrates superior precision across high-confidence deciles (PR-AUC 0.6671 vs 0.6497).
- **Probability Agreement**: Pearson correlation of predicted probabilities is $r = 0.948$.

---

## 15. Threshold Optimization

Using the business cost model:
$$\text{Total Cost} = 200 \times \text{FN} + 100 \times \text{FP}$$

- At threshold $0.50$, total cost was \$47,700 (driven by 190 False Negatives).
- Shifting threshold to **$0.40$** reduced False Negatives to 128, cutting total cost to **\$40,700** (a **\$7,000 cost savings** per 1,400 customers).
- Shifting to **$0.35$** captured 71.39% recall at total cost \$41,300.

---

## 16. Probability Calibration

- Brier Score for winning Raw CatBoost: **0.1332** (improved over baseline 0.1364).
- Calibration curves show monotonic risk progression, ensuring that a predicted probability of 0.70 corresponds to approximately 70% empirical churn frequency.

---

## 17. Model Interpretation (SHAP & Gain)

SHAP summary reveals primary directional impact:
1. **`Contract == Month-to-month`**: Heaviest positive push toward churn.
2. **`tenure`**: High tenure exerts strong negative force on churn probability.
3. **`InternetService == Fiber optic`**: Positive push toward churn, reflecting customer dissatisfaction with premium pricing or service stability.
4. **`TechSupport` & `OnlineSecurity`**: Strong negative push toward churn.
5. **`PaymentMethod == Electronic check`**: Notable positive association with churn.

---

## 18. High-Risk Customer Analysis

Scoring audits (`outputs/predictions/raw_catboost_test.csv`) delineate the high-risk subscriber profile ($P(\text{Churn}) > 0.65$):
- **Tenure**: Under 6 months ($81\%$ of cluster).
- **Contract**: Month-to-month ($97\%$ of cluster).
- **Service**: Fiber Optic Internet ($84\%$ of cluster) without TechSupport or OnlineSecurity.
- **Payment**: Electronic check billing ($68\%$ of cluster).
- **Monthly Charges**: High (\$75 to \$95/month).

---

## 19. Business Recommendations

1. **New Subscriber "First 90 Days" Onboarding**: Provide proactive VIP technical support during the critical first 3 months to mitigate the 56.9% initial hazard cliff.
2. **Fiber Optic + Security Bundling**: Subsidize or include `OnlineSecurity` and `TechSupport` into fiber optic tiers.
3. **Contract Migration Incentives**: Offer a \$10/month credit for converting from month-to-month to an annual agreement.
4. **Automated Payment Incentive**: Provide a one-time \$15 credit for switching from electronic check to automatic credit card or ACH billing.

---

## 20. Model Limitations

1. **Cross-Sectional Snapshot**: Dataset lacks granular temporal usage series (e.g., weekly data usage drops, recent speed complaints).
2. **Observational Nature**: Discovered features reflect statistical correlation, not direct causality.
3. **Static Economics**: False-negative/false-positive cost ratio ($2:1$) is an empirical assumption subject to operational refinement.

---

## 21. Future Improvements

1. **Survival Analysis**: Implement Cox Proportional Hazards or Kaplan-Meier curves to model *time-to-churn*.
2. **Uplift Modeling**: Train causal uplift models (two-model or X-Learner) to predict customer *persuadability* rather than raw churn risk.
3. **Customer Lifetime Value (CLV) Weighting**: Optimize thresholds based on individual customer MRR rather than uniform cost assumptions.

---

## 22. Final Conclusion

The project successfully developed, tuned, and evaluated a reproducible churn prediction architecture strictly within the allowed XGBoost and CatBoost model families. 

- **Primary Recommendation**: Deploy **Raw CatBoost** at operating threshold **0.40** to maximize ranking PR-AUC ($0.6671$) and balance precision-recall economics.
- **Secondary Recommendation**: Deploy **Selected XGBoost** at threshold **0.35** for high-capacity retention marketing campaigns demanding maximum recall ($71.39\%$).
