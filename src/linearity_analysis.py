import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def binned_churn_rates(frame: pd.DataFrame, feature: str, target: str = "Churn", bins: int = 10) -> pd.DataFrame:
    values = frame[[feature, target]].dropna().copy()
    values["bin"] = pd.qcut(values[feature], q=bins, duplicates="drop")
    return values.groupby("bin", observed=True)[target].agg(customer_count="size", churn_rate="mean").reset_index()


def mutual_information(frame: pd.DataFrame, features: list[str], target: str = "Churn", random_state: int = 42) -> pd.DataFrame:
    values = frame[features].copy()
    values = pd.get_dummies(values, dummy_na=True).fillna(0)
    scores = mutual_info_classif(values, frame[target], random_state=random_state)
    return pd.DataFrame({"feature": values.columns, "mutual_information": scores}).sort_values("mutual_information", ascending=False)


def diagnose_relationship(rate_table: pd.DataFrame) -> str:
    rates = rate_table["churn_rate"].to_numpy()
    if len(rates) < 3:
        return "insufficient data"
    changes = np.diff(rates)
    if np.all(changes >= 0) or np.all(changes <= 0):
        return "monotonic; assess whether nonlinear or approximately linear on the logit scale"
    return "non-monotonic or threshold-driven"
