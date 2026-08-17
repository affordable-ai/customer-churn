from xgboost import XGBClassifier

from .config import RANDOM_STATE


def baseline_xgboost(**overrides) -> XGBClassifier:
    params = dict(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    params.update(overrides)
    return XGBClassifier(**params)


def tuned_xgboost(**overrides) -> XGBClassifier:
    params = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_alpha=0.1, reg_lambda=2.0, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    params.update(overrides)
    return XGBClassifier(**params)
