"""
tests/test_predict.py
---------------------
Basic unit tests. Run: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_clean_text():
    from preprocessing import clean_text
    result = clean_text("URGENT!!! Easy $5000 from http://scam.com")
    assert "urgent" in result
    assert "http" not in result
    assert "$" not in result


def test_suspicious_keywords():
    from feature_engineering import SUSPICIOUS_KEYWORDS
    assert len(SUSPICIOUS_KEYWORDS) > 0
    assert "urgent hiring" in SUSPICIOUS_KEYWORDS


def test_structured_features_shape():
    from preprocessing import preprocess_dataframe
    from feature_engineering import build_structured_features
    import pandas as pd

    df = pd.DataFrame([{
        "title": "Engineer", "description": "Build stuff",
        "company_profile": "Big corp", "requirements": "Python",
        "benefits": "Health insurance", "salary_range": "60k",
        "telecommuting": 0, "has_company_logo": 1,
    }])
    df = preprocess_dataframe(df)
    feats = build_structured_features(df)
    assert feats.shape[0] == 1
    assert feats.shape[1] > 0
