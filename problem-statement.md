# Advanced End-to-End Telco Customer Churn ML Project

You are an expert **Data Scientist, Statistical Analyst, Machine Learning Engineer, and ML Architect**.

Develop a complete, rigorous, reproducible **Customer Churn Prediction project for a telecommunications company** using Python locally.

The project must combine:

1. Business understanding
2. Data-quality analysis
3. Exploratory Data Analysis
4. Statistical hypothesis testing
5. Linearity vs nonlinearity analysis
6. Dependency and association analysis
7. Multicollinearity analysis
8. Comprehensive feature engineering
9. Comprehensive feature selection
10. XGBoost modeling
11. CatBoost modeling
12. Hyperparameter optimization
13. Cross-validation
14. Model evaluation
15. Probability calibration/threshold analysis
16. Model interpretation
17. Business recommendations

Do NOT use Logistic Regression, Random Forest, SVM, Neural Networks, LightGBM, or any other predictive model.

The only predictive models allowed are:

* XGBoost
* CatBoost

---

# 1. Business Problem

The telecommunications company wants to predict whether a customer is likely to churn.

The target variable is:

```text
Churn
```

where:

```text
Yes = Customer churned: 10 - 9 Recall 99%, Precission:
No  = Customer retained
```

This is a binary classification problem.

The objective is not merely to maximize accuracy.

The solution should answer:

* Which customers are most likely to churn?
* What characteristics are associated with churn?
* Are the relationships between features and churn linear or nonlinear?
* Which features provide the most predictive information?
* Which engineered features improve predictive performance?
* Which model performs better: XGBoost or CatBoost?
* What is the optimal probability threshold?
* What business segments should receive retention attention?
* What are the limitations of the model?

---

# 2. Dataset

Use:

**Telco Customer Churn Dataset**

Source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

Expected file:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset must be downloaded manually and stored under:

```text
data/raw/
```

Do not depend on the Kaggle API for the core project.

---

# 3. Strict Modeling Requirement

Only these two predictive algorithms may be used:

## Model 1

XGBoost:

```python
XGBClassifier
```

## Model 2

CatBoost:

```python
CatBoostClassifier
```

Do not introduce another classifier as a baseline.

Instead, establish a **simple XGBoost configuration as the baseline** and compare it with:

* tuned XGBoost
* tuned CatBoost

The final comparison should therefore include:

```text
Baseline XGBoost
       ↓
Tuned XGBoost
       ↓
Tuned CatBoost
```

The purpose is to understand:

* baseline performance
* effect of hyperparameter optimization
* effect of algorithm choice

---

# 4. Project Structure

Create the following structure:

```text
telco-customer-churn/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │
│   ├── interim/
│   │   └── cleaned_telco.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_validation.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_validation.csv
│       └── y_test.csv
│
├── notebooks/
│   │
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_statistical_testing.ipynb
│   ├── 04_linearity_nonlinearity_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_feature_selection.ipynb
│   ├── 07_xgboost_baseline.ipynb
│   ├── 08_xgboost_tuning.ipynb
│   ├── 09_catboost_tuning.ipynb
│   ├── 10_model_evaluation.ipynb
│   ├── 11_threshold_optimization.ipynb
│   └── 12_model_interpretation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   │
│   ├── eda.py
│   ├── statistical_tests.py
│   ├── linearity_analysis.py
│   │
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   │
│   ├── xgboost_model.py
│   ├── catboost_model.py
│   ├── hyperparameter_tuning.py
│   │
│   ├── evaluation.py
│   ├── threshold_optimization.py
│   ├── calibration.py
│   ├── interpretation.py
│   │
│   ├── prediction.py
│   └── utils.py
│
├── models/
│   ├── baseline_xgboost/
│   ├── tuned_xgboost/
│   └── tuned_catboost/
│
├── outputs/
│   ├── tables/
│   ├── predictions/
│   ├── statistical_tests/
│   └── feature_selection/
│
├── reports/
│   ├── figures/
│   ├── statistical_report.md
│   ├── feature_engineering_report.md
│   ├── model_comparison.md
│   └── telco_churn_report.md
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_feature_selection.py
│   └── test_prediction.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

# 5. Stage 1 — Data Understanding

Load the dataset and perform a complete data audit.

Generate:

* shape
* columns
* data types
* numerical features
* categorical features
* target distribution
* unique values
* cardinality
* missing values
* duplicated rows
* constant features
* suspicious values
* invalid values
* outliers

Create a data dictionary:

| Feature | Type | Meaning | Missing | Cardinality | Treatment |
| ------- | ---- | ------- | ------: | ----------: | --------- |

Investigate `TotalCharges` carefully because it may contain blank values even though it represents a numerical quantity.

Convert it safely to numeric.

Investigate:

```text
customerID
```

and explain why it should generally not be used as a predictive feature.

---

# 6. Stage 2 — Exploratory Data Analysis

Perform comprehensive EDA.

## Target

Analyze:

* churn count
* churn percentage
* class imbalance

Plot:

* count plot
* percentage plot

Calculate:

```text
Churn Rate
Retention Rate
Imbalance Ratio
```

---

# 7. Numerical EDA

Analyze:

```text
tenure
MonthlyCharges
TotalCharges
```

Calculate:

* mean
* median
* variance
* standard deviation
* min
* max
* quartiles
* IQR
* skewness
* kurtosis

Visualize:

* histogram
* KDE
* boxplot
* violin plot
* ECDF where useful

Compare churn vs non-churn distributions.

---

# 8. Categorical EDA

Analyze churn rates for:

* gender
* SeniorCitizen
* Partner
* Dependents
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract
* PaperlessBilling
* PaymentMethod

Always distinguish:

```text
Customer Count
```

from:

```text
Churn Rate
```

because a category with many customers can have a large churn count without having the highest churn risk.

---

# 9. Statistical Testing Framework

Create a dedicated statistical analysis module.

The objective is NOT to run every statistical test blindly.

Choose tests based on:

* variable type
* distribution
* sample size
* assumptions
* research question

Document why each test is selected.

---

# 10. Normality Testing

For numerical features investigate distributional assumptions.

Use:

### Shapiro-Wilk

For smaller samples or appropriate subsets.

### Anderson-Darling

As an additional normality assessment.

### D'Agostino K²

For larger samples.

Also use:

* Q-Q plots
* skewness
* kurtosis

Important:

Do NOT conclude that a variable is “linear” or “nonlinear” based on normality.

Normality and linearity are different concepts.

Explicitly document this distinction.

---

# 11. Numerical Feature vs Churn Tests

Because churn is binary, compare numerical variables between:

```text
Churn = Yes
```

and:

```text
Churn = No
```

Use appropriate tests.

### Independent t-test

When assumptions are reasonably satisfied.

Test:

```text
H0: Mean values are equal
H1: Mean val ues are different
```

### Welch's t-test

Use when equal variance cannot be assumed.

### Mann-Whitney U

Use as a non-parametric alternative.

Report:

* test statistic
* p-value
* effect size
* interpretation

Do not report p-values without effect sizes.

---

# 12. Effect Sizes

Calculate meaningful effect sizes.

For numerical variables consider:

* Cohen's d
* rank-biserial correlation

For categorical variables consider:

* Cramér's V
* Phi coefficient for 2×2 tables

Classify effect sizes as:

* negligible
* small
* moderate
* large

Use appropriate conventions and explain them.

---

# 13. Categorical Statistical Tests

For categorical variables use:

### Chi-square test of independence

Test:

```text
H0:
Feature and churn are independent.

H1:
Feature and churn are associated.
```

Calculate:

* chi-square statistic
* degrees of freedom
* p-value
* Cramér's V

When expected cell counts are too small, use an appropriate alternative such as Fisher's exact test for suitable contingency tables.

Create a statistical-testing summary table:

| Feature | Test | Statistic | P-value | Effect Size | Significant? |
| ------- | ---- | --------: | ------: | ----------: | ------------ |

---

# 14. Multiple Hypothesis Testing

Because many statistical tests will be performed, control for multiple comparisons.

Implement:

### Benjamini-Hochberg FDR correction

Report:

* original p-value
* adjusted p-value
* significance status

Do not simply perform dozens of hypothesis tests and interpret raw p-values.

---

# 15. Correlation Analysis

For numerical variables calculate:

### Pearson correlation

Measures linear association.

### Spearman correlation

Measures monotonic association and is more robust to non-normal distributions/outliers.

### Kendall's Tau

Use where appropriate for rank-based association.

Create:

* Pearson correlation matrix
* Spearman correlation matrix

Compare Pearson vs Spearman.

If:

```text
Pearson is low
Spearman is high
```

investigate whether the relationship may be nonlinear but monotonic.

Do not conclude nonlinearity solely from this comparison.

---

# 16. Nonlinear Dependence Analysis

Use additional techniques to identify nonlinear relationships.

Calculate:

### Mutual Information

Estimate how much information a feature provides about churn.

Use:

```python
mutual_info_classif
```

Rank features based on mutual information.

Explain:

* MI detects dependency beyond simple linear correlation
* MI does not establish causality
* MI scores should be interpreted comparatively

Also investigate nonlinear relationships using:

* binned churn rates
* partial dependence
* accumulated local effects where useful
* model-based feature importance

---

# 17. Linearity vs Nonlinearity Analysis

This is a critical section.

Do not simply say:

> “XGBoost performs better, therefore the data is nonlinear.”

Instead perform multiple analyses.

Investigate whether continuous variables have approximately linear relationships with the **log-odds of churn**.

For numerical features:

```text
tenure
MonthlyCharges
TotalCharges
```

perform:

### Binning analysis

Divide each numerical feature into quantile bins.

Calculate churn rate per bin.

Plot:

```text
Feature Value
      vs
Churn Rate
```

Look for:

* monotonicity
* thresholds
* U-shapes
* inverted U-shapes
* abrupt changes
* saturation

---

# 18. Linearity-of-Logit Diagnostics

Investigate whether numerical variables have an approximately linear relationship with the logit of churn probability.

Use transformations and diagnostics such as:

* log transformation
* polynomial terms
* spline-based diagnostic analysis
* Box-Tidwell-style assessment where appropriate

Do not use an additional predictive model as part of the final model comparison.

These techniques are diagnostic/statistical analyses only.

Explain whether:

```text
Feature → Churn relationship
```

appears:

* approximately linear
* monotonic but nonlinear
* strongly nonlinear
* threshold-driven
* interaction-dependent

---

# 19. Polynomial / Transformation Analysis

For continuous variables investigate transformations such as:

```text
x
log(x)
sqrt(x)
x²
x³
```

where mathematically meaningful.

Do not automatically include all transformations.

Determine whether transformations improve:

* association with target
* monotonicity
* predictive information
* stability

Document whether each transformation is:

```text
Useful
Not Useful
Potentially Redundant
```

---

# 20. Interaction Analysis

Investigate whether churn depends on combinations of variables.

Examples to investigate:

```text
Contract × Tenure
Contract × MonthlyCharges
InternetService × TechSupport
InternetService × OnlineSecurity
SeniorCitizen × Contract
PaymentMethod × Contract
Tenure × MonthlyCharges
```

Do not assume these interactions exist.

Use:

* grouped churn-rate analysis
* statistical interaction diagnostics
* XGBoost interaction importance
* CatBoost feature interaction analysis

Document meaningful interactions.

---

# 21. Multicollinearity Analysis

Although tree-based models are less sensitive to multicollinearity than linear models, still investigate redundancy.

Calculate:

### Pearson correlation

### Spearman correlation

### Variance Inflation Factor

VIF should be used as a diagnostic, not as an automatic deletion rule.

Identify:

* highly correlated feature pairs
* redundant features
* derived variables containing overlapping information

Explain whether redundant variables should be:

* retained
* removed
* transformed

---

# 22. Feature Engineering

Create a comprehensive but controlled feature-engineering framework.

Potential features include:

## Tenure features

```text
tenure_months
tenure_years
is_new_customer
is_long_term_customer
tenure_bucket
```

## Billing features

```text
avg_monthly_charge
charge_per_tenure_month
total_charge_consistency
high_monthly_charge
```

Do not create mathematically invalid ratios.

Handle zero denominators.

---

# 23. Service Aggregation Features

Create:

```text
number_of_services
number_of_security_services
number_of_streaming_services
number_of_support_services
```

based on logically related service columns.

Document exactly which columns contribute to each aggregate.

---

# 24. Customer Profile Features

Create meaningful combinations such as:

```text
has_dependents
has_partner_or_dependents
senior_single_customer
```

Only retain features that have business meaning and do not duplicate existing information unnecessarily.

---

# 25. Contract Features

Engineer:

```text
is_month_to_month
is_one_year_contract
is_two_year_contract
```

where useful.

Investigate whether contract type interacts with:

```text
tenure
monthly charges
services
```

---

# 26. Digital Engagement Features

Consider aggregating:

```text
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
```

into meaningful service-engagement indicators.

Avoid creating features that simply duplicate the original variables without adding information.

---

# 27. Feature Transformation

Evaluate:

* log transformation
* square-root transformation
* winsorization where justified
* clipping
* binning
* quantile transformation
* rank transformation

Do not automatically apply transformations to tree-based models.

Test whether they provide measurable benefit.

---

# 28. Feature Selection Framework

Implement feature selection in four categories.

## A. Filter Methods

Use:

### Missing-value filtering

Remove features with excessive missingness.

### Variance filtering

Identify near-zero variance features.

### Correlation filtering

Remove redundant features where appropriate.

### Statistical significance

Use:

* chi-square
* Mann-Whitney
* t-test
* effect size

### Mutual Information

Rank features by target dependency.

---

# 29. B. Wrapper Methods

Use wrapper-style selection where computationally reasonable.

Consider:

### Recursive Feature Elimination

But the wrapper must use only the allowed model families:

```text
XGBoost
CatBoost
```

Do not introduce Logistic Regression or Random Forest merely to perform RFE.

Because the dataset is relatively small, compare computational cost against benefit.

---

# 30. C. Embedded Feature Selection

Use:

### XGBoost feature importance

Compare:

* gain
* weight
* cover

Prefer gain-based importance when discussing predictive contribution.

### CatBoost feature importance

Investigate appropriate CatBoost importance measures.

CatBoost also provides native feature-selection functionality through `select_features`, which can be incorporated into the feature-selection experiment.

---

# 31. D. Permutation-Based Selection

Use:

### Permutation Importance

Evaluate how model performance changes when each feature is randomly permuted.

Perform permutation importance on validation data or through cross-validation.

Do not compute importance on the test set and then use it for feature selection.

---

# 32. SHAP-Based Feature Selection

Use SHAP to investigate:

* global importance
* local importance
* feature direction
* nonlinear effects
* interactions

For each model produce:

```text
SHAP Summary
SHAP Bar Plot
SHAP Dependence Plot
```

Use SHAP as an interpretation/selection aid rather than blindly dropping every feature with a small SHAP value.

---

# 33. Feature Selection Stability

This is important.

Do not select features from a single train/test split and assume the result is stable.

Repeat feature selection across cross-validation folds where practical.

Calculate:

```text
Feature Selection Frequency
```

For example:

| Feature        | Selected in CV folds | Selection Frequency |
| -------------- | -------------------: | ------------------: |
| tenure         |                  5/5 |                100% |
| Contract       |                  5/5 |                100% |
| MonthlyCharges |                  4/5 |                 80% |

Identify:

* stable features
* unstable features
* potentially redundant features

---

# 34. Feature Selection Decision Framework

Create several feature sets:

```text
Feature Set A
Raw features

Feature Set B
Raw + engineered features

Feature Set C
Statistically selected features

Feature Set D
Model-selected features

Feature Set E
Final optimized feature set
```

Compare model performance across feature sets.

Do not assume that fewer features means a better model.

The final feature set should balance:

* performance
* stability
* interpretability
* computational efficiency
* business relevance

---

# 35. Data Leakage Prevention

This is mandatory.

All target-dependent feature engineering and feature selection must be performed without using the test set.

Be particularly careful with:

* target encoding
* mutual information
* statistical tests
* target-based aggregations
* feature selection
* SHAP-based selection
* threshold optimization

For any target-dependent operation, fit/derive it using training data only within the model-development workflow.

The test set must remain untouched until final evaluation.

---

# 36. Data Splitting

Use:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

Prefer an internal validation strategy or cross-validation on the training set.

Final test set:

```text
20%
```

must remain completely unseen.

---

# 37. Baseline XGBoost

Build a simple baseline:

```python
XGBClassifier(...)
```

Use a reasonable configuration without extensive tuning.

Record:

* training performance
* cross-validation performance
* validation performance
* test performance

The baseline exists to establish a reference point.

---

# 38. XGBoost Hyperparameter Optimization

Tune appropriate parameters.

Consider:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma
reg_alpha
reg_lambda
```

Use:

```text
RandomizedSearchCV
```

or another defensible search strategy.

Do not perform an unnecessarily enormous grid search.

Use:

```text
StratifiedKFold
```

for cross-validation.

Optimize based on a metric aligned with the business problem.

Consider:

```text
PR-AUC
ROC-AUC
F1
Recall
```

Do not automatically optimize accuracy.

---

# 39. CatBoost Model

Build:

```python
CatBoostClassifier(...)
```

Use CatBoost's native categorical feature capability where appropriate.

Do NOT unnecessarily one-hot encode categorical variables before CatBoost. CatBoost's documentation specifically notes that categorical features can be passed directly through `cat_features`, and warns against conventional preprocessing with one-hot encoding in this context.

Tune:

```text
iterations
depth
learning_rate
l2_leaf_reg
random_strength
bagging_temperature
border_count
```

Also investigate:

```text
class_weights
```

if appropriate for churn imbalance.

Use early stopping where appropriate.

---

# 40. XGBoost vs CatBoost Data Representation

Explicitly document the preprocessing difference.

For XGBoost:

```text
categorical → numerical encoding
```

For CatBoost:

```text
categorical → native categorical representation
```

Compare whether native categorical handling provides an advantage.

Do not assume that CatBoost will outperform XGBoost.

Let the empirical results determine the conclusion.

---

# 41. Cross-Validation

Use:

```text
StratifiedKFold
```

with approximately:

```text
5 folds
```

Calculate for each model:

* mean ROC-AUC
* standard deviation ROC-AUC
* mean PR-AUC
* standard deviation PR-AUC
* mean F1
* mean recall
* mean precision

Example:

| Model            | ROC-AUC Mean | ROC-AUC Std | PR-AUC Mean | F1 Mean |
| ---------------- | -----------: | ----------: | ----------: | ------: |
| Baseline XGBoost |              |             |             |         |
| Tuned XGBoost    |              |             |             |         |
| Tuned CatBoost   |              |             |             |         |

---

# 42. Model Comparison

Compare:

```text
Baseline XGBoost
Tuned XGBoost
Tuned CatBoost
```

Metrics:

* Accuracy
* Precision
* Recall
* Specificity
* F1
* ROC-AUC
* PR-AUC
* Log Loss
* Brier Score
* confusion matrix

Do not use accuracy as the primary selection criterion.

---

# 43. Statistical Comparison of Models

Where practical, statistically compare model predictions.

Use appropriate paired procedures.

Consider:

### McNemar's Test

Compare classification errors between two classifiers on the same test observations.

### Bootstrap Confidence Intervals

Estimate confidence intervals for:

* ROC-AUC
* PR-AUC
* F1
* Recall
* Precision

For ROC-AUC comparison, consider an appropriate paired method such as DeLong's test if implemented correctly.

Clearly distinguish:

```text
Statistical significance
```

from:

```text
Practical significance
```

---

# 44. Threshold Optimization

Do not automatically use:

```text
threshold = 0.50
```

Evaluate thresholds such as:

```text
0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
```

For every threshold calculate:

* precision
* recall
* F1
* specificity
* number of customers flagged
* percentage of customer base flagged

Create a threshold-performance chart.

---

# 45. Business Cost-Based Threshold

Create an optional cost framework.

Define:

```text
Cost(False Negative)
Cost(False Positive)
```

For example:

```text
FN = lost customer / lost revenue
FP = unnecessary retention campaign cost
```

Calculate expected business cost:

```text
Total Cost =
FN × Cost_FN
+
FP × Cost_FP
```

Find the threshold that minimizes expected cost.

Clearly state that these costs are assumptions unless supplied by the business.

---

# 46. Probability Calibration

Evaluate whether predicted churn probabilities are reliable.

Use:

* reliability diagram
* calibration curve
* Brier score
* log loss

Investigate whether the model saying:

```text
80% churn probability
```

actually corresponds approximately to an 80% observed churn rate.

Do not calibrate automatically unless necessary.

If calibration is applied, perform it without contaminating the test set.

---

# 47. Model Interpretation

For XGBoost:

Analyze:

* gain
* feature importance
* permutation importance
* SHAP

For CatBoost:

Analyze:

* feature importance
* permutation importance
* SHAP

CatBoost provides native feature importance and feature-selection functionality, which should be considered where appropriate.

---

# 48. SHAP Analysis

For the winning model generate:

### Global explanation

Top features influencing churn.

### Local explanation

Explain individual customers.

### Dependence analysis

Investigate how:

```text
tenure
MonthlyCharges
Contract
TotalCharges
```

affect churn predictions.

### Interaction analysis

Investigate important feature interactions.

Explicitly explain:

> SHAP explains model behavior and predictive contribution; it does not establish causality.

---

# 49. Customer Risk Segmentation

Generate:

```text
customerID
actual_churn
predicted_churn
churn_probability
risk_category
```

Example:

```text
Low Risk
Medium Risk
High Risk
```

Do not treat arbitrary probability thresholds as universally correct.

Document the chosen threshold methodology.

---

# 50. High-Risk Customer Analysis

For high-risk customers investigate:

* contract
* tenure
* monthly charges
* services
* payment method
* internet service
* support services

Create a profile of high-risk customers.

The analysis should answer:

> What does a typical high-risk churn customer look like?

---

# 51. Business Segmentation

Identify segments with:

* high churn rate
* high predicted churn probability
* high customer value if revenue information is available

If revenue is unavailable, do not invent customer lifetime value.

Explain that the next version should combine:

```text
Churn Probability
×
Customer Value
```

to prioritize retention.

---

# 52. Business Recommendations

Translate the results into actions.

Examples:

* target month-to-month customers
* improve onboarding for new customers
* identify customers with high monthly charges
* provide proactive technical support
* investigate payment-method friction
* offer contract incentives where appropriate

Do not state:

> “This feature causes churn.”

Instead say:

> “Customers with this characteristic showed higher predicted churn risk/observed churn rates.”

Recommend controlled experiments to determine whether interventions actually reduce churn.

---

# 53. Model Limitations

Discuss:

* dataset size
* sample representativeness
* historical bias
* missing behavioral information
* lack of customer-level temporal history
* possible data drift
* class imbalance
* probability calibration
* feature selection instability
* causal limitations
* fairness considerations
* assumptions behind cost-sensitive thresholding

---

# 54. Final Model Selection Framework

Do NOT simply select:

> Model with highest ROC-AUC.

Use a decision framework:

```text
Predictive Performance
        +
Recall
        +
Precision
        +
PR-AUC
        +
Probability Calibration
        +
Feature Stability
        +
Interpretability
        +
Business Cost
        +
Operational Complexity
```

Then select the final model.

For example:

```text
Final Model = CatBoost
```

is acceptable only if the experimental evidence supports it.

Likewise:

```text
Final Model = XGBoost
```

is acceptable if it performs better or provides a better business tradeoff.

---

# 55. Final Experimental Matrix

Create a comprehensive experiment table:

| Experiment | Features   | Model    | Tuning | ROC-AUC | PR-AUC |   F1 | Recall | Precision |
| ---------- | ---------- | -------- | ------ | ------: | -----: | ---: | -----: | --------: |
| E1         | Raw        | XGBoost  | No     |         |        |      |        |           |
| E2         | Engineered | XGBoost  | No     |         |        |      |        |           |
| E3         | Selected   | XGBoost  | No     |         |        |      |        |           |
| E4         | Selected   | XGBoost  | Yes    |         |        |      |        |           |
| E5         | Raw        | CatBoost | No     |         |        |      |        |           |
| E6         | Engineered | CatBoost | No     |         |        |      |        |           |
| E7         | Selected   | CatBoost | Yes    |         |        |      |        |           |

This table should demonstrate whether:

```text
Feature Engineering
```

and:

```text
Feature Selection
```

actually improve performance.

---

# 56. Final Feature Selection Report

Create a table:

| Feature | Statistical Evidence |   MI | Correlation | XGB Importance | CatBoost Importance | SHAP | Stability | Final Decision |
| ------- | -------------------- | ---: | ----------: | -------------: | ------------------: | ---: | --------: | -------------- |

This becomes one of the most important analytical outputs of the project.

---

# 57. Statistical Conclusions

Create a final statistical summary:

```text
Feature
Association Test
P-value
Adjusted P-value
Effect Size
Linear/Nonlinear Evidence
Predictive Importance
Conclusion
```

Example:

| Feature        | Association | Nonlinear Evidence  | Effect   | Prediction |
| -------------- | ----------- | ------------------- | -------- | ---------- |
| tenure         | Significant | Strong              | Moderate | High       |
| Contract       | Significant | Interaction         | Large    | High       |
| MonthlyCharges | Significant | Nonlinear/threshold | Moderate | High       |

Only populate results based on actual analysis.

Never fabricate numbers.

---

# 58. Required Visualizations

Generate at minimum:

## Data Understanding

* missing-value plot
* cardinality plot
* target distribution

## EDA

* numerical distributions
* boxplots
* churn-rate plots
* categorical churn-rate plots

## Statistics

* Q-Q plots
* Pearson heatmap
* Spearman heatmap
* correlation comparison
* effect-size plots
* statistical significance summary

## Nonlinearity

* feature vs churn-rate bins
* feature transformation comparisons
* nonlinear dependence/MI ranking
* interaction plots

## Feature Selection

* MI ranking
* XGBoost importance
* CatBoost importance
* permutation importance
* SHAP importance
* feature stability

## Model Evaluation

* ROC curve
* Precision-Recall curve
* confusion matrices
* calibration curves
* threshold analysis
* model comparison

---

# 59. Final Report Structure

Create:

```text
reports/telco_churn_report.md
```

with:

# Executive Summary

# 1. Business Problem

# 2. Dataset

# 3. Data Quality

# 4. Exploratory Data Analysis

# 5. Statistical Testing

# 6. Linearity vs Nonlinearity

# 7. Feature Engineering

# 8. Feature Selection

# 9. XGBoost Baseline

# 10. XGBoost Optimization

# 11. CatBoost Optimization

# 12. Cross-Validation

# 13. Model Evaluation

# 14. Statistical Model Comparison

# 15. Threshold Optimization

# 16. Probability Calibration

# 17. Model Interpretation

# 18. High-Risk Customer Analysis

# 19. Business Recommendations

# 20. Model Limitations

# 21. Future Improvements

# 22. Final Conclusion

---

# 60. README

Create a comprehensive README containing:

## Project Overview

## Business Problem

## Dataset

## Architecture

## Project Structure

## Environment Setup

## Installation

## Running the Project

## Running Notebooks

## Running Tests

## Statistical Methodology

## Feature Engineering

## Feature Selection

## XGBoost

## CatBoost

## Evaluation Metrics

## Model Selection Criteria

## Results

## Business Interpretation

## Limitations

## Future Improvements

---

# 61. Requirements

At minimum consider:

```text
pandas
numpy
scipy
scikit-learn
xgboost
catboost
matplotlib
seaborn
statsmodels
shap
joblib
pytest
jupyter
```

Use additional libraries only when justified.

---

# 62. Code Quality

Use:

* modular functions
* type hints where useful
* docstrings
* logging
* configuration management
* reproducible random seeds
* exception handling
* unit tests

Avoid:

* duplicated code
* huge notebook cells
* hardcoded paths
* data leakage
* unnecessary abstractions

---

# 63. Reproducibility

Use:

```python
RANDOM_STATE = 42
```

where appropriate.

Save:

* trained models
* selected features
* feature metadata
* hyperparameters
* evaluation results
* predictions

Use:

```text
models/
outputs/
```

for artifacts.

---

# 64. Critical Rule About Results

Never fabricate results.

If the dataset is unavailable:

```text
Provide executable code.
```

If the dataset is available:

```text
Actually execute the analysis.
```

Report actual:

* sample sizes
* churn percentage
* statistical results
* selected features
* XGBoost metrics
* CatBoost metrics
* optimal threshold
* final model

---

# 65. Final Deliverable

The finished project should demonstrate the following reasoning:

```text
                     BUSINESS PROBLEM
                           │
                           ▼
                  DATA UNDERSTANDING
                           │
                           ▼
                    DATA QUALITY
                           │
                           ▼
                         EDA
                           │
                           ▼
              STATISTICAL HYPOTHESIS TESTING
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       LINEARITY TESTS             NONLINEARITY
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 FEATURE ENGINEERING
                           │
                           ▼
                  FEATURE SELECTION
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
          XGBOOST                     CATBOOST
             │                           │
             ▼                           ▼
       HYPERPARAMETER              HYPERPARAMETER
          TUNING                       TUNING
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  CROSS VALIDATION
                           │
                           ▼
                   MODEL EVALUATION
                           │
                           ▼
             STATISTICAL MODEL COMPARISON
                           │
                           ▼
                 THRESHOLD OPTIMIZATION
                           │
                           ▼
                  MODEL INTERPRETATION
                           │
                           ▼
                  CUSTOMER RISK SCORING
                           │
                           ▼
                 BUSINESS RECOMMENDATION
```

The final project should demonstrate not just:

> “I trained XGBoost and CatBoost.”

It should demonstrate:

> **“I statistically investigated the structure of the data, identified associations and nonlinear patterns, engineered and selected features systematically, trained and optimized two appropriate nonlinear ensemble models, statistically compared them, optimized the decision threshold based on business costs, interpreted the predictions, and translated the results into actionable churn-management recommendations.”**

This is the expected quality level for a strong **Data Scientist / ML Engineer interview or academic project**.
