import pandas as pd
from src.feature_engineering import engineer_features


def test_feature_engineering_protects_zero_tenure():
    frame = pd.DataFrame({'tenure': [0], 'TotalCharges': [0.0], 'MonthlyCharges': [50.0], 'SeniorCitizen': [0], 'Partner': ['No'], 'Dependents': ['No'], 'Contract': ['Month-to-month'], 'OnlineSecurity': ['No'], 'OnlineBackup': ['No'], 'DeviceProtection': ['No'], 'TechSupport': ['No'], 'StreamingTV': ['No'], 'StreamingMovies': ['No'], 'PhoneService': ['Yes'], 'MultipleLines': ['No']})
    result = engineer_features(frame)
    assert result['avg_monthly_charge'].isna().all()
    assert result['number_of_services'].iloc[0] >= 1


def test_feature_engineering_creates_interactions_and_ratios():
    frame = pd.DataFrame({
        'tenure': [12],
        'TotalCharges': [600.0],
        'MonthlyCharges': [50.0],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'Contract': ['Month-to-month'],
        'InternetService': ['Fiber optic'],
        'PaymentMethod': ['Electronic check'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['No'],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
    })
    result = engineer_features(frame)
    assert 'contract_internetservice' in result.columns
    assert result['contract_internetservice'].iloc[0] == 'Month-to-month__Fiber optic'
    assert 'charge_per_service' in result.columns
    assert result['charge_per_service'].iloc[0] > 0
    assert result['fiber_no_support'].iloc[0] == 1
    assert result['auto_payment'].iloc[0] == 0

