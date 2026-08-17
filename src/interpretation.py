import pandas as pd


def xgboost_importance(model) -> pd.DataFrame:
    booster = model.get_booster()
    return pd.DataFrame([{"feature": key, "gain": value} for key, value in booster.get_score(importance_type="gain").items()]).sort_values("gain", ascending=False)


def catboost_importance(model, feature_names) -> pd.DataFrame:
    return pd.DataFrame({"feature": feature_names, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
