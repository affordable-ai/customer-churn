import pandas as pd
from sklearn.inspection import permutation_importance


class FeatureSelector:
    """Own the notebook-derived logical feature sets used in model experiments."""

    SELECTED_FEATURES = [
        "tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
        "OnlineSecurity", "TechSupport", "PaymentMethod", "MultipleLines",
        "PaperlessBilling", "StreamingMovies", "SeniorCitizen", "Dependents",
        "is_month_to_month", "is_two_year_contract", "is_new_customer",
        "avg_monthly_charge", "charge_per_service", "charge_delta",
        "total_charge_consistency", "number_of_services", "number_of_streaming_services",
        "contract_internetservice", "contract_payment", "fiber_no_security",
        "fiber_no_support", "has_tech_and_security", "auto_payment",
    ]

    def select(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return notebook-supported logical features available in a frame."""
        columns = [column for column in self.SELECTED_FEATURES if column in frame.columns]
        return frame.loc[:, columns].copy()



def permutation_rank(model, X, y, scoring="average_precision", random_state=42) -> pd.DataFrame:
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=10, random_state=random_state, n_jobs=-1)
    names = getattr(X, "columns", [f"feature_{i}" for i in range(X.shape[1])])
    return pd.DataFrame({"feature": names, "importance_mean": result.importances_mean, "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)


def selection_frequency(rankings: list[pd.DataFrame], top_k: int = 20) -> pd.DataFrame:
    counts = {}
    for ranking in rankings:
        for feature in ranking.head(top_k)["feature"]:
            counts[feature] = counts.get(feature, 0) + 1
    total = len(rankings)
    return pd.DataFrame([{"feature": feature, "selected_folds": count, "selection_frequency": count / total} for feature, count in counts.items()]).sort_values("selection_frequency", ascending=False)
