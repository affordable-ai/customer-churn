import pandas as pd
from src.data_validation import audit_dataframe, validate_frame


def test_audit_detects_blank_total_charges():
    frame = pd.DataFrame({'customerID': ['a'], 'TotalCharges': [None], 'Churn': [1]})
    audit = audit_dataframe(frame)
    assert audit['blank_total_charges'] == 1


def test_validate_rejects_empty_frame():
    try:
        validate_frame(pd.DataFrame())
    except ValueError:
        return
    raise AssertionError('empty frames must be rejected')
