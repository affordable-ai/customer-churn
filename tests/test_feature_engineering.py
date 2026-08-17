import pandas as pd
from src.feature_engineering import engineer_features


def test_feature_engineering_protects_zero_tenure():
    frame = pd.DataFrame({'tenure': [0], 'TotalCharges': [0.0], 'MonthlyCharges': [50.0], 'SeniorCitizen': [0], 'Partner': ['No'], 'Dependents': ['No'], 'Contract': ['Month-to-month'], 'OnlineSecurity': ['No'], 'OnlineBackup': ['No'], 'DeviceProtection': ['No'], 'TechSupport': ['No'], 'StreamingTV': ['No'], 'StreamingMovies': ['No'], 'PhoneService': ['Yes'], 'MultipleLines': ['No']})
    result = engineer_features(frame)
    assert result['avg_monthly_charge'].isna().all()
    assert result['number_of_services'].iloc[0] >= 1
