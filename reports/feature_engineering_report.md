# Feature Engineering & Selection Report

## 1. Executive Summary & Design Principles

Feature engineering in this project transforms static customer snapshot records into dynamic behavioral signals while adhering to strict **leakage-prevention principles**:
- **Zero Target Leakage**: No target-encoding, out-of-fold target statistics, or target-dependent aggregations were used.
- **Split-Safe Parameterization**: Continuous thresholding (e.g., training median for `MonthlyCharges` = \$70.35) was computed strictly on `X_train` and mapped into validation and test sets.
- **Mathematical Safety**: Ratios such as `TotalCharges / tenure` explicitly safeguard against division-by-zero for new customers (`tenure = 0`) by imputing nulls with training medians.

---

## 2. Engineered Feature Architecture

A total of 19 domain-driven features were created across 5 functional groupings:

### A. Tenure Dynamics
1. `tenure_months`: Baseline continuous months active.
2. `tenure_years`: Continuous tenure scaled to annual increments (`tenure / 12.0`).
3. `is_new_customer`: Binary indicator for high-hazard onboarding phase (`tenure <= 6` months). Churn hazard in this window exceeds 50%.
4. `is_long_term_customer`: Binary indicator for loyal, established subscribers (`tenure >= 48` months). Churn hazard drops below 10%.

### B. Billing Consistency & Pricing Friction
5. `avg_monthly_charge`: Historical average monthly cost (`TotalCharges / tenure`). Divergence from `MonthlyCharges` indicates recent price hikes or promotional discounts expiring.
6. `charge_per_tenure_month`: Intensity ratio measuring price sensitivity over customer lifecycle (`MonthlyCharges / tenure`).
7. `total_charge_consistency`: Residual billing difference (`TotalCharges - (MonthlyCharges * tenure)`). Detects unbilled charges, sudden fee adjustments, or mid-cycle plan upgrades.
8. `high_monthly_charge`: Binary indicator for charges above the training median (\$70.35).

### C. Service Aggregation & Breadth of Adoption
9. `number_of_security_services`: Count of protective products (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`).
10. `number_of_streaming_services`: Count of entertainment products (`StreamingTV`, `StreamingMovies`).
11. `number_of_support_services`: Specialized technical support access (`TechSupport`).
12. `number_of_services`: Comprehensive breadth metric summing all 8 potential service subscriptions. Customers with $\ge 4$ services exhibit vastly reduced churn propensity.
13. `digital_engagement_count`: Total digital value-added services adopted.

### D. Customer Demographics & Family Profile
14. `has_dependents`: Binary flag for children/dependents.
15. `has_partner_or_dependents`: Composite family anchor indicator. Families have substantially higher switching friction and lower churn.
16. `senior_single_customer`: Vulnerable/isolated customer segment (`SeniorCitizen == 1`, no partner, no dependents).

### E. Contractual Commitment Flags
17. `is_month_to_month`: Primary churn driver indicator (`Contract == "Month-to-month"`).
18. `is_one_year_contract`: Moderate commitment flag.
19. `is_two_year_contract`: High-loyalty commitment flag.

---

## 3. Feature Selection Methodology

To identify the most parsimonious and predictive feature subset, multiple selection paradigms were synthesized:
1. **Filter Methods**: Mutual Information (MI) and Cramér's V ranking.
2. **Embedded Tree Importance**: XGBoost `gain` and CatBoost native prediction values change (PVC).
3. **Model-Agnostic Permutation Importance**: Evaluating drop in PR-AUC when feature values are randomly shuffled on development data.

### Selected Feature Subset (23 Features)
The resulting selected subset retains the highest-signal raw and engineered variables:
- **Core Numerics**: `tenure`, `MonthlyCharges`, `TotalCharges`, `tenure_months`
- **Billing Metrics**: `avg_monthly_charge`, `charge_per_tenure_month`, `total_charge_consistency`
- **Core Categoricals**: `Contract`, `InternetService`, `OnlineSecurity`, `TechSupport`, `PaymentMethod`, `MultipleLines`, `PaperlessBilling`, `StreamingMovies`, `PhoneService`
- **Demographics & Profile**: `SeniorCitizen`, `Dependents`, `has_dependents`
- **Contract & Service Encodings**: `is_month_to_month`, `is_one_year_contract`, `is_two_year_contract`, `number_of_streaming_services`

---

## 4. Empirical Performance Across Feature Representations

Evaluating the three feature representations (Raw vs. Engineered vs. Selected) across both XGBoost and CatBoost yields the following held-out test results:

| Feature Representation | Model Family | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw** | **CatBoost** | **0.6760** | **0.6671** | **0.8443** | 0.5885 | 0.6578 | 0.6212 |
| **Selected** | **CatBoost** | 0.6752 | 0.6584 | 0.8455 | 0.5947 | 0.6551 | 0.6234 |
| **Engineered** | **CatBoost** | 0.6751 | 0.6639 | 0.8463 | 0.5727 | 0.7059 | 0.6323 |
| **Raw** | **XGBoost** | 0.6729 | 0.6609 | 0.8451 | 0.5976 | 0.6551 | 0.6250 |
| **Selected** | **XGBoost** | 0.6725 | 0.6497 | 0.8460 | 0.5681 | **0.7139** | **0.6327** |
| **Engineered** | **XGBoost** | 0.6720 | 0.6492 | 0.8419 | 0.5714 | 0.6310 | 0.5997 |

### Engineering & Selection Findings:
1. **CatBoost Native Efficiency on Raw Features**: CatBoost natively constructs optimal target-based combinations of categorical features during tree building. Consequently, raw categorical inputs allow CatBoost to maximize overall ranking precision (**Test PR-AUC 0.6671**), outperforming manual one-hot engineering.
2. **Selected Features Boost XGBoost Recall**: For XGBoost, using the `selected` feature set paired with a cost-optimal threshold of 0.35 achieves the highest test recall (**71.39%**) and highest test F1 (**0.6327**).
3. **Dimensionality vs. Overfitting**: Adding all 19 engineered features to XGBoost slightly reduced test PR-AUC (0.6492 vs 0.6609 for raw), showing that tree algorithms can suffer slight variance inflation when collinear ratios (`charge_per_tenure_month`) are added without aggressive pruning.
