import pandas as pd
from src.feature_selection import FeatureSelector, selection_frequency


def test_selection_frequency_counts_top_features():
    rankings = [pd.DataFrame({'feature': ['a', 'b']}), pd.DataFrame({'feature': ['a', 'c']})]
    result = selection_frequency(rankings, top_k=1)
    assert result.iloc[0]['feature'] == 'a'
    assert result.iloc[0]['selection_frequency'] == 1.0


def test_feature_selector_selects_relevant_features():
    selector = FeatureSelector()
    frame = pd.DataFrame({'tenure': [10], 'gender': ['Male'], 'charge_per_service': [25.0], 'noise': [999]})
    selected = selector.select(frame)
    assert 'tenure' in selected.columns
    assert 'charge_per_service' in selected.columns
    assert 'gender' not in selected.columns
    assert 'noise' not in selected.columns

