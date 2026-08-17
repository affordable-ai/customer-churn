import pandas as pd


def numeric_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    summary = frame[columns].describe().T
    summary["variance"] = frame[columns].var()
    summary["skewness"] = frame[columns].skew()
    summary["kurtosis"] = frame[columns].kurtosis()
    summary["iqr"] = summary["75%"] - summary["25%"]
    return summary


def categorical_churn_rates(frame: pd.DataFrame, columns: list[str], target: str = "Churn") -> pd.DataFrame:
    return frame.groupby(columns, dropna=False)[target].agg(customer_count="size", churn_rate="mean").reset_index()
