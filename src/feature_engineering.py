"""
feature_engineering.py
-----------------------
Builds structured + NLP features from the cleaned DataFrame.
Outputs a feature matrix X and target vector y.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix
import joblib
import os

# Suspicious keywords that often appear in fake postings
SUSPICIOUS_KEYWORDS = [
    "urgent hiring", "quick money", "easy income", "no interview",
    "registration fee", "work from home", "guaranteed income",
    "no experience needed", "earn fast", "wire transfer",
    "immediate joining", "part time easy", "data entry"
]


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric/boolean features from raw columns.
    These complement TF-IDF text features.
    """
    feats = pd.DataFrame()

    # Text length features
    feats["desc_length"]    = df["description"].apply(len)
    feats["title_length"]   = df["title"].apply(len)
    feats["profile_length"] = df["company_profile"].apply(len)

    # Missing info indicators (common in fake postings)
    feats["missing_salary"]          = (df.get("salary_range", pd.Series([""] * len(df))).fillna("") == "").astype(int)
    feats["missing_company_profile"] = (df["company_profile"] == "").astype(int)
    feats["missing_requirements"]    = (df.get("requirements", pd.Series([""] * len(df))).fillna("") == "").astype(int)

    # Remote job flag
    if "telecommuting" in df.columns:
        feats["is_remote"] = df["telecommuting"].fillna(0).astype(int)
    else:
        feats["is_remote"] = 0

    # Has company logo
    if "has_company_logo" in df.columns:
        feats["has_logo"] = df["has_company_logo"].fillna(0).astype(int)
    else:
        feats["has_logo"] = 0

    # Suspicious keyword count in description
    feats["suspicious_keyword_count"] = df["description"].apply(
        lambda txt: sum(kw in txt for kw in SUSPICIOUS_KEYWORDS)
    )

    return feats


def build_features(df: pd.DataFrame, tfidf=None, fit=True):
    """
    Combine TF-IDF features with structured features.

    Parameters
    ----------
    df    : cleaned DataFrame
    tfidf : existing TfidfVectorizer (for inference); None = create new one
    fit   : True during training, False during inference

    Returns
    -------
    X_combined : sparse matrix ready for model input
    tfidf      : fitted vectorizer (save this for inference)
    """
    # Combine all text columns into one string per row
    df["combined_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["company_profile"] + " " +
        df.get("requirements", pd.Series([""] * len(df))).fillna("")
    )

    if fit:
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        X_text = tfidf.fit_transform(df["combined_text"])
    else:
        X_text = tfidf.transform(df["combined_text"])

    # Structured features
    X_struct = csr_matrix(build_structured_features(df).values.astype(np.float32))

    # Stack text + structured features side by side
    X_combined = hstack([X_text, X_struct])

    return X_combined, tfidf


if __name__ == "__main__":
    print("Feature engineering module loaded OK.")
    print(f"Suspicious keywords tracked: {len(SUSPICIOUS_KEYWORDS)}")
