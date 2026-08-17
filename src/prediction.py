import numpy as np
import pandas as pd


def score_customers(model, features: pd.DataFrame, customer_ids: pd.Series, actual=None, threshold: float = 0.5) -> pd.DataFrame:
    probabilities = model.predict_proba(features)[:, 1]
    result = pd.DataFrame({"customerID": customer_ids.to_numpy(), "churn_probability": probabilities, "predicted_churn": (probabilities >= threshold).astype(int)})
    if actual is not None:
        result["actual_churn"] = np.asarray(actual)
    result["risk_category"] = pd.cut(result["churn_probability"], bins=[-np.inf, 0.30, 0.60, np.inf], labels=["Low Risk", "Medium Risk", "High Risk"])
    return result
