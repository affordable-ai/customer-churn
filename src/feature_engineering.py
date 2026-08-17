import numpy as np
import pandas as pd

YES = {"Yes", "yes", 1, True}


class FeatureEngineer:
    """Create documented, target-independent churn features."""

    FEATURE_GROUPS = {
        "tenure": ["tenure_months", "tenure_years", "is_new_customer", "is_long_term_customer"],
        "billing": ["avg_monthly_charge", "charge_per_tenure_month", "total_charge_consistency", "high_monthly_charge"],
        "services": ["number_of_services", "number_of_security_services", "number_of_streaming_services", "number_of_support_services", "digital_engagement_count"],
        "profile": ["has_dependents", "has_partner_or_dependents", "senior_single_customer"],
        "contract": ["is_month_to_month", "is_one_year_contract", "is_two_year_contract"],
    }

    def transform(self, frame: pd.DataFrame, monthly_charge_threshold: float | None = None) -> pd.DataFrame:
        """Add business-defined features without target-dependent calculations."""
        result = frame.copy()
        result["tenure_months"] = result["tenure"]
        result["tenure_years"] = result["tenure"] / 12.0
        result["is_new_customer"] = (result["tenure"] <= 6).astype(int)
        result["is_long_term_customer"] = (result["tenure"] >= 48).astype(int)
        result["avg_monthly_charge"] = result["TotalCharges"] / result["tenure"].replace(0, np.nan)
        result["charge_per_tenure_month"] = result["MonthlyCharges"] / result["tenure"].replace(0, np.nan)
        result["total_charge_consistency"] = result["TotalCharges"] - (result["MonthlyCharges"] * result["tenure"])
        threshold = monthly_charge_threshold if monthly_charge_threshold is not None else result["MonthlyCharges"].median()
        result["high_monthly_charge"] = (result["MonthlyCharges"] >= threshold).astype(int)
        service_groups = {
            "number_of_security_services": ["OnlineSecurity", "OnlineBackup", "DeviceProtection"],
            "number_of_streaming_services": ["StreamingTV", "StreamingMovies"],
            "number_of_support_services": ["TechSupport"],
        }
        for name, columns in service_groups.items():
            result[name] = sum(_yes(result[column]) for column in columns if column in result)
        service_columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        result["number_of_services"] = sum(_yes(result[column]) for column in service_columns if column in result)
        result["has_dependents"] = _yes(result["Dependents"])
        result["has_partner_or_dependents"] = ((_yes(result["Partner"]) + _yes(result["Dependents"])) > 0).astype(int)
        result["senior_single_customer"] = ((result["SeniorCitizen"] == 1) & (result["Partner"] == "No") & (result["Dependents"] == "No")).astype(int)
        result["is_month_to_month"] = (result["Contract"] == "Month-to-month").astype(int)
        result["is_one_year_contract"] = (result["Contract"] == "One year").astype(int)
        result["is_two_year_contract"] = (result["Contract"] == "Two year").astype(int)
        result["digital_engagement_count"] = sum(_yes(result[column]) for column in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"])
        return result.replace([np.inf, -np.inf], np.nan)


def _yes(series: pd.Series) -> pd.Series:
    return series.isin(YES).astype(int)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add business-defined features without target-dependent calculations."""
    return FeatureEngineer().transform(frame)
