from typing import Sequence
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_xgboost_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def prepare_catboost_frame(frame: pd.DataFrame, categorical_columns: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Keep categoricals as strings for CatBoost native categorical handling."""
    result = frame.copy()
    columns = list(categorical_columns or result.select_dtypes(exclude="number").columns)
    for column in columns:
        result[column] = result[column].fillna("Missing").astype(str)
    for column in result.select_dtypes(include="number"):
        result[column] = result[column].fillna(result[column].median())
    return result, [result.columns.get_loc(column) for column in columns]
