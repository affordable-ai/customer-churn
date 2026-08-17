from pathlib import Path
import pandas as pd

from .config import ID_COLUMN, RAW_DATA_PATH, TARGET


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and minimally clean the Kaggle Telco dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    if TARGET in frame:
        frame[TARGET] = frame[TARGET].map({"Yes": 1, "No": 0}).astype("Int64")
    return frame


def split_features_target(frame: pd.DataFrame):
    """Return identifiers, predictors, and binary target without using customerID."""
    required = {ID_COLUMN, TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[ID_COLUMN].copy(), frame.drop(columns=[ID_COLUMN, TARGET]).copy(), frame[TARGET].astype(int).copy()
