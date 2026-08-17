import pandas as pd
from src.feature_selection import selection_frequency


def test_selection_frequency_counts_top_features():
    rankings = [pd.DataFrame({'feature': ['a', 'b']}), pd.DataFrame({'feature': ['a', 'c']})]
    result = selection_frequency(rankings, top_k=1)
    assert result.iloc[0]['feature'] == 'a'
    assert result.iloc[0]['selection_frequency'] == 1.0
