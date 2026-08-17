import numpy as np
import pandas as pd
from src.prediction import score_customers


class ProbabilityModel:
    def predict_proba(self, features):
        return np.array([[0.8, 0.2], [0.1, 0.9]])


def test_score_customers_creates_risk_categories():
    result = score_customers(ProbabilityModel(), pd.DataFrame({'x': [1, 2]}), pd.Series(['a', 'b']), actual=[0, 1], threshold=0.5)
    assert result['predicted_churn'].tolist() == [0, 1]
    assert result['risk_category'].tolist() == ['Low Risk', 'High Risk']
