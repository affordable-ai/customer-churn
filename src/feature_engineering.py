import numpy as np
import pandas as pd

YES = {"Yes", "yes", 1, True}


class FeatureEngineer:
    """Create documented, target-independent churn features."""

    FEATURE_GROUPS = {
        "tenure": ["tenure_months", "tenure_years", "is_new_customer", "is_long_term_customer"],
        "billing": ["avg_monthly_charge", "charge_per_tenure_month", "total_charge_consistency", "high_monthly_charge", "charge_per_service", "charge_delta"],
        "services": ["number_of_services", "number_of_security_services", "number_of_streaming_services", "number_of_support_services", "digital_engagement_count", "has_tech_and_security", "fiber_no_support", "fiber_no_security"],
        "profile": ["has_dependents", "has_partner_or_dependents", "senior_single_customer", "auto_payment"],
        "contract": ["is_month_to_month", "is_one_year_contract", "is_two_year_contract"],
        "interactions": ["contract_internetservice", "contract_payment", "internetservice_payment"],
    }

    def transform(self, frame: pd.DataFrame, monthly_charge_threshold: float | None = None) -> pd.DataFrame:
        """Add business-defined features without target-dependent calculations."""
        result = frame.copy()
        total_charges = pd.to_numeric(result["TotalCharges"], errors="coerce") if "TotalCharges" in result else pd.Series(np.nan, index=result.index)

        result["tenure_months"] = result["tenure"]
        result["tenure_years"] = result["tenure"] / 12.0
        result["is_new_customer"] = (result["tenure"] <= 6).astype(int)
        result["is_long_term_customer"] = (result["tenure"] >= 48).astype(int)

        # Financial / billing ratios
        result["avg_monthly_charge"] = total_charges / result["tenure"].replace(0, np.nan)
        result["charge_per_tenure_month"] = result["MonthlyCharges"] / result["tenure"].replace(0, np.nan)
        result["total_charge_consistency"] = total_charges - (result["MonthlyCharges"] * result["tenure"])
        result["charge_delta"] = result["MonthlyCharges"] - result["avg_monthly_charge"]

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
        result["charge_per_service"] = result["MonthlyCharges"] / (result["number_of_services"] + 1)

        result["has_dependents"] = _yes(result["Dependents"]) if "Dependents" in result else 0
        result["has_partner_or_dependents"] = ((_yes(result["Partner"]) + _yes(result["Dependents"])) > 0).astype(int) if "Partner" in result and "Dependents" in result else 0
        result["senior_single_customer"] = ((result["SeniorCitizen"] == 1) & (result["Partner"] == "No") & (result["Dependents"] == "No")).astype(int) if all(c in result for c in ["SeniorCitizen", "Partner", "Dependents"]) else 0

        if "Contract" in result:
            result["is_month_to_month"] = (result["Contract"] == "Month-to-month").astype(int)
            result["is_one_year_contract"] = (result["Contract"] == "One year").astype(int)
            result["is_two_year_contract"] = (result["Contract"] == "Two year").astype(int)

        result["digital_engagement_count"] = sum(_yes(result[column]) for column in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"] if column in result)

        # High-signal interaction and cohort flags
        if "Contract" in result and "InternetService" in result:
            result["contract_internetservice"] = result["Contract"].astype(str) + "__" + result["InternetService"].astype(str)
        if "Contract" in result and "PaymentMethod" in result:
            result["contract_payment"] = result["Contract"].astype(str) + "__" + result["PaymentMethod"].astype(str)
        if "InternetService" in result and "PaymentMethod" in result:
            result["internetservice_payment"] = result["InternetService"].astype(str) + "__" + result["PaymentMethod"].astype(str)

        if "OnlineSecurity" in result and "TechSupport" in result:
            result["has_tech_and_security"] = ((result["OnlineSecurity"] == "Yes") & (result["TechSupport"] == "Yes")).astype(int)
        if "InternetService" in result and "TechSupport" in result:
            result["fiber_no_support"] = ((result["InternetService"] == "Fiber optic") & (result["TechSupport"] == "No")).astype(int)
        if "InternetService" in result and "OnlineSecurity" in result:
            result["fiber_no_security"] = ((result["InternetService"] == "Fiber optic") & (result["OnlineSecurity"] == "No")).astype(int)
        if "PaymentMethod" in result:
            result["auto_payment"] = result["PaymentMethod"].isin(["Bank transfer (automatic)", "Credit card (automatic)"]).astype(int)

        return result.replace([np.inf, -np.inf], np.nan)


def _yes(series: pd.Series) -> pd.Series:
    return series.isin(YES).astype(int)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add business-defined features without target-dependent calculations."""
    return FeatureEngineer().transform(frame)

