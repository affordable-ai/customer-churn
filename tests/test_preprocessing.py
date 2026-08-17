import pandas as pd
from src.preprocessing import make_xgboost_preprocessor, prepare_catboost_frame


def test_xgboost_preprocessor_handles_categories():
    frame = pd.DataFrame({'number': [1, None], 'category': ['a', 'b']})
    transformed = make_xgboost_preprocessor(frame).fit_transform(frame)
    assert transformed.shape[0] == 2


def test_catboost_keeps_categoricals_as_strings():
    result, indices = prepare_catboost_frame(pd.DataFrame({'category': [None, 'a'], 'number': [1, None]}))
    assert result['category'].tolist() == ['Missing', 'a']
    assert indices == [0]
