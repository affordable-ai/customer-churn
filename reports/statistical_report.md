# Telco Customer Churn — Comprehensive Statistical & Linearity Report

## 1. Overview & Statistical Framework

This statistical report evaluates all numerical and categorical features in the Telco Customer Churn dataset with respect to the binary target variable `Churn` (0 = Retained, 1 = Churned). 

### Methodological Guardrails
- **Leakage Prevention**: All statistical tests, effect sizes, p-values, and non-linearity diagnostics are strictly computed on the training split ($N = 4,507$) to avoid data leakage into the validation or test sets.
- **Hypothesis Testing Corrections**: Benjamini-Hochberg False Discovery Rate (FDR) adjustment ($\alpha = 0.05$) is applied to control the expected proportion of false positives across multiple hypothesis tests.
- **Causality vs. Association**: All findings reflect statistical associations within observational data and must not be interpreted as definitive causal relationships without controlled A/B testing.

---

## 2. Numerical Feature Analysis & Hypothesis Testing

Because numerical variables (`tenure`, `MonthlyCharges`, `TotalCharges`) exhibit non-normal distributions (confirmed via Shapiro-Wilk testing and visual Q-Q plots), non-parametric **Mann-Whitney U** tests are conducted alongside standardized effect sizes (**Cohen's d**).

| Feature | Statistical Test | Test Statistic ($U$) | Raw $p$-value | Cohen's $d$ | FDR Adjusted $p$-value | Statistically Significant ($\alpha=0.05$) | Direction of Association |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tenure`** | Mann-Whitney U | $1,042,784.0$ | $1.47 \times 10^{-130}$ | **-0.880** | $4.42 \times 10^{-130}$ | **Yes** | Strong negative (newer customers churn far more) |
| **`TotalCharges`** | Mann-Whitney U | $1,384,597.5$ | $3.22 \times 10^{-53}$ | **-0.485** | $4.84 \times 10^{-53}$ | **Yes** | Moderate negative (driven heavily by low tenure) |
| **`MonthlyCharges`** | Mann-Whitney U | $2,465,303.5$ | $2.62 \times 10^{-36}$ | **+0.472** | $2.62 \times 10^{-36}$ | **Yes** | Moderate positive (higher monthly bill correlates with churn) |

### Key Observations:
1. **Tenure Effect**: Possesses the largest effect size ($|d| = 0.88$, considered a large effect in Cohen's benchmarks). New customers are in an acute hazard window.
2. **Monthly Charges Effect**: Customers who churn pay an average of \$18.60 more per month than retained customers (\$74.44 vs \$55.84).
3. **Total Charges Interaction**: While TotalCharges is lower for churners due to short tenure, the ratio of TotalCharges to tenure reveals pricing consistency.

---

## 3. Categorical Feature Association Analysis

Contingency table analysis is performed using the **Pearson Chi-Square ($\chi^2$) test of independence**, with **Cramér's V** quantifying the magnitude of associative strength ($V \in [0, 1]$).

| Rank | Feature | Test | $\chi^2$ Statistic | Degrees of Freedom | Raw $p$-value | Cramér's V | FDR Adjusted $p$-value | Significance |
| :---: | :--- | :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| 1 | **`Contract`** | $\chi^2$ Contingency | 768.68 | 2 | $1.21 \times 10^{-167}$ | **0.4130** | $1.94 \times 10^{-166}$ | Significant |
| 2 | **`OnlineSecurity`** | $\chi^2$ Contingency | 549.58 | 2 | $4.58 \times 10^{-120}$ | **0.3492** | $3.67 \times 10^{-119}$ | Significant |
| 3 | **`TechSupport`** | $\chi^2$ Contingency | 532.69 | 2 | $2.13 \times 10^{-116}$ | **0.3438** | $1.14 \times 10^{-115}$ | Significant |
| 4 | **`InternetService`** | $\chi^2$ Contingency | 477.93 | 2 | $1.66 \times 10^{-104}$ | **0.3256** | $6.63 \times 10^{-104}$ | Significant |
| 5 | **`PaymentMethod`** | $\chi^2$ Contingency | 406.51 | 3 | $8.60 \times 10^{-88}$ | **0.3003** | $2.75 \times 10^{-87}$ | Significant |
| 6 | **`OnlineBackup`** | $\chi^2$ Contingency | 388.96 | 2 | $3.45 \times 10^{-85}$ | **0.2938** | $9.20 \times 10^{-85}$ | Significant |
| 7 | **`DeviceProtection`** | $\chi^2$ Contingency | 351.53 | 2 | $4.64 \times 10^{-77}$ | **0.2793** | $1.06 \times 10^{-76}$ | Significant |
| 8 | **`StreamingMovies`** | $\chi^2$ Contingency | 234.46 | 2 | $1.23 \times 10^{-51}$ | **0.2281** | $2.45 \times 10^{-51}$ | Significant |
| 9 | **`StreamingTV`** | $\chi^2$ Contingency | 232.98 | 2 | $2.57 \times 10^{-51}$ | **0.2274** | $4.57 \times 10^{-51}$ | Significant |
| 10 | **`PaperlessBilling`** | $\chi^2$ Contingency | 159.56 | 1 | $1.41 \times 10^{-36}$ | **0.1882** | $2.26 \times 10^{-36}$ | Significant |
| 11 | **`Partner`** | $\chi^2$ Contingency | 100.82 | 1 | $9.82 \times 10^{-24}$ | **0.1496** | $1.43 \times 10^{-23}$ | Significant |
| 12 | **`Dependents`** | $\chi^2$ Contingency | 89.28 | 1 | $3.43 \times 10^{-21}$ | **0.1407** | $4.58 \times 10^{-21}$ | Significant |
| 13 | **`SeniorCitizen`** | $\chi^2$ Contingency | 74.05 | 1 | $7.63 \times 10^{-18}$ | **0.1282** | $9.39 \times 10^{-18}$ | Significant |
| 14 | **`MultipleLines`** | $\chi^2$ Contingency | 7.92 | 2 | 0.0191 | **0.0419** | 0.0218 | Significant |
| 15 | **`PhoneService`** | $\chi^2$ Contingency | 0.74 | 1 | 0.3884 | **0.0128** | 0.4143 | Not Significant |
| 16 | **`gender`** | $\chi^2$ Contingency | 0.28 | 1 | 0.5973 | **0.0079** | 0.5973 | Not Significant |

### Insights on Categorical Variables:
- **Contract Type**: The dominant predictor ($V = 0.4130$). Month-to-month contracts have a churn rate of **~43%**, compared to **~11%** for one-year and **~3%** for two-year contracts.
- **Service Add-ons**: Lack of `OnlineSecurity` and `TechSupport` strongly associate with churn ($V \approx 0.34$–$0.35$). These utility features act as "customer adhesives".
- **Demographics**: `gender` ($p = 0.597$) and `PhoneService` ($p = 0.388$) show zero statistically significant association with churn after FDR adjustment.

---

## 4. Linearity vs. Non-Linearity Diagnostics

### Linearity-of-Logit Analysis

A logistic model assumes that continuous features relate linearly to the log-odds ($\text{logit}(p) = \ln(p / (1-p))$) of churn. Quantile binning reveals that this assumption fails:

#### 1. `tenure` (Monotonic but Non-Linear Logit)
- Quantile 1 (0 to 2.25 months): Churn rate = **56.91%**, Log-odds = **+0.278**
- Quantile 2 (2.25 to 9.0 months): Churn rate = **42.81%**, Log-odds = **-0.290**
- Quantile 3 (9.0 to 17.0 months): Churn rate = **33.01%**, Log-odds = **-0.708**
- Quantile 5 (29.0 to 42.0 months): Churn rate = **19.54%**, Log-odds = **-1.415**
- Quantile 8 (67.0 to 72.0 months): Churn rate = **4.55%**, Log-odds = **-3.043**
- *Diagnostic*: The decline in log-odds is steep during the first 12 months ($\Delta \text{log-odds} = -0.99$) and flattens out thereafter.

#### 2. `MonthlyCharges` (Distinctly Non-Monotonic & Threshold-Driven)
- \$18.40 – \$20.30: Churn rate = **8.70%** (Basic landline service, very low churn)
- \$20.30 – \$55.00: Churn rate = **14.3% – 28.0%**
- \$70.35 – \$100.15: Churn rate peaks at **39.75%** (Fiber optic service without security)
- \$100.15 – \$118.75: Churn rate drops to **26.88%** (Fully bundled premium customers)
- *Diagnostic*: **Strong non-linearity and non-monotonicity**. A linear or logistic model cannot capture this hump-shaped relationship without complex polynomial or spline terms, giving tree-based ensembles (XGBoost/CatBoost) a decisive advantage.

---

## 5. Mutual Information (MI) Ranking

Mutual Information measures non-parametric dependency (capturing both linear and non-linear interactions):

| Feature Sub-attribute | Mutual Information Score |
| :--- | :---: |
| `Contract_Month-to-month` | **0.0977** |
| `OnlineSecurity_No` | **0.0730** |
| `tenure` | **0.0708** |
| `TechSupport_No` | **0.0654** |
| `Contract_Two year` | **0.0636** |
| `MonthlyCharges` | **0.0501** |
| `PaymentMethod_Electronic check` | **0.0461** |
| `InternetService_Fiber optic` | **0.0432** |
| `TotalCharges` | **0.0412** |
| `OnlineBackup_No` | **0.0391** |

---

## 6. Statistical Conclusions for Modeling Strategy

1. **Why Tree Ensembles**: The severe non-linearity of `MonthlyCharges` and the threshold drop of `tenure` confirm why linear baselines (like Logistic Regression) are structurally handicapped.
2. **Key Signals to Preserve**: The model must capture the interplay between `Contract`, `tenure`, `MonthlyCharges`, and support services (`TechSupport`, `OnlineSecurity`).
3. **Irrelevant Features**: `gender` and `PhoneService` contribute virtually zero information and can be pruned during feature selection without loss of fidelity.
