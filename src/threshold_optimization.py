import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def threshold_table(y_true, probabilities, thresholds=None, cost_fn=1.0, cost_fp=1.0) -> pd.DataFrame:
    thresholds = thresholds or np.arange(0.20, 0.71, 0.05)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    rows = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        tn = int(((y_true == 0) & (predicted == 0)).sum())
        fp = int(((y_true == 0) & (predicted == 1)).sum())
        fn = int(((y_true == 1) & (predicted == 0)).sum())
        rows.append({"threshold": float(threshold), "precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0), "f1": f1_score(y_true, predicted, zero_division=0), "specificity": tn / (tn + fp) if tn + fp else 0.0, "flagged_count": int(predicted.sum()), "flagged_pct": float(predicted.mean()), "expected_cost": fn * cost_fn + fp * cost_fp})
    return pd.DataFrame(rows)


def select_cost_threshold(table: pd.DataFrame) -> float:
    return float(table.loc[table["expected_cost"].idxmin(), "threshold"])
