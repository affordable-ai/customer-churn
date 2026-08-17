import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def numerical_tests(frame: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        yes = frame.loc[frame[target] == 1, column].dropna().to_numpy()
        no = frame.loc[frame[target] == 0, column].dropna().to_numpy()
        statistic, p_value = stats.mannwhitneyu(yes, no, alternative="two-sided")
        pooled = np.sqrt((yes.var(ddof=1) + no.var(ddof=1)) / 2)
        effect = (yes.mean() - no.mean()) / pooled if pooled else 0.0
        rows.append({"feature": column, "test": "Mann-Whitney U", "statistic": statistic, "p_value": p_value, "effect_size_cohens_d": effect})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_p_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
        result["significant_fdr_05"] = result["adjusted_p_value"] < 0.05
    return result


def categorical_tests(frame: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        table = pd.crosstab(frame[column], frame[target])
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        effect = np.sqrt(chi2 / n) if n else 0.0
        rows.append({"feature": column, "test": "Chi-square", "statistic": chi2, "degrees_of_freedom": dof, "p_value": p_value, "cramers_v": effect})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_p_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
        result["significant_fdr_05"] = result["adjusted_p_value"] < 0.05
    return result
