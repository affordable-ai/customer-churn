from typing import Any
import pandas as pd

from .config import ID_COLUMN, TARGET


def audit_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Create a reproducible data-quality audit."""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    outliers = {}
    for column in numeric:
        values = frame[column].dropna()
        if values.empty:
            outliers[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[column] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    return {
        "shape": frame.shape,
        "columns": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_values": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1],
        "cardinality": frame.nunique(dropna=False).to_dict(),
        "target_distribution": frame[TARGET].value_counts(dropna=False).to_dict() if TARGET in frame else {},
        "outlier_counts": outliers,
        "blank_total_charges": int((frame["TotalCharges"].isna()).sum()) if "TotalCharges" in frame else 0,
        "identifier_unique": bool(frame[ID_COLUMN].is_unique) if ID_COLUMN in frame else None,
    }


def validate_frame(frame: pd.DataFrame) -> None:
    """Fail early on invalid target and identifier conditions."""
    if frame.empty:
        raise ValueError("Dataset is empty")
    if TARGET not in frame or ID_COLUMN not in frame:
        raise ValueError(f"Dataset must include {TARGET!r} and {ID_COLUMN!r}")
    if frame[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    if not set(frame[TARGET].unique()).issubset({0, 1, "Yes", "No"}):
        raise ValueError("Target must contain only Yes/No or 1/0 values")
