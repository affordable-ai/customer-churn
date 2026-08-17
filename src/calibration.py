import pandas as pd
from sklearn.calibration import calibration_curve


def calibration_table(y_true, probabilities, bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": predicted, "observed_churn_rate": observed})
