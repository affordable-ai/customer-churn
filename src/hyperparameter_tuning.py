from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import RANDOM_STATE


def randomized_tune(estimator, parameter_distributions, X, y, n_iter=15):
    search = RandomizedSearchCV(estimator, parameter_distributions, n_iter=n_iter, scoring="average_precision", cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    return search.fit(X, y)
