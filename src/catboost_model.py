from catboost import CatBoostClassifier

from .config import RANDOM_STATE


def tuned_catboost(**overrides) -> CatBoostClassifier:
    params = dict(iterations=500, depth=6, learning_rate=0.04, l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0, border_count=64, loss_function="Logloss", eval_metric="AUC", verbose=False, random_seed=RANDOM_STATE, thread_count=-1, allow_writing_files=False)
    params.update(overrides)
    return CatBoostClassifier(**params)
