"""
predict.py
----------
Loads the saved model + vectorizer and runs inference on new job postings.
Used by both the FastAPI backend and Streamlit app.
"""

import joblib
import pandas as pd
import numpy as np
import os

from preprocessing import preprocess_dataframe, clean_text
from feature_engineering import build_features, SUSPICIOUS_KEYWORDS

MODELS_DIR = r"C:\Projects\fraud-job-detection\models"

# Load once at import time
_model = None
_tfidf = None


def _load_artifacts():
    global _model, _tfidf
    if _model is None:
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Run 'python src/train.py' first."
            )
        _model = joblib.load(model_path)
        _tfidf = joblib.load(tfidf_path)


def _build_reasons(job: dict, fraud_prob: float) -> list:
    """Generate human-readable risk reasons."""
    reasons = []

    if not job.get("company_profile", "").strip():
        reasons.append("Missing company profile")

    if not job.get("salary_range", "").strip():
        reasons.append("No salary range provided")

    desc = job.get("description", "").lower()
    found_kw = [kw for kw in SUSPICIOUS_KEYWORDS if kw in desc]
    if found_kw:
        reasons.append(f"Suspicious keywords: {', '.join(found_kw[:3])}")

    if not job.get("has_company_logo", 1):
        reasons.append("No company logo")

    if fraud_prob > 0.85:
        reasons.append("Very high fraud probability score")
    elif fraud_prob > 0.6:
        reasons.append("Elevated fraud probability score")

    return reasons if reasons else ["No specific red flags detected"]


def predict_single(job: dict) -> dict:
    """
    Predict fraud probability for a single job posting.

    Parameters
    ----------
    job : dict with keys matching dataset columns
          (title, description, company_profile, salary_range, ...)

    Returns
    -------
    dict with prediction, fraud_probability, risk_level, reasons
    """
    _load_artifacts()

    df = pd.DataFrame([job])
    df = preprocess_dataframe(df)
    X, _ = build_features(df, tfidf=_tfidf, fit=False)

    fraud_prob = float(_model.predict_proba(X)[0][1])
    prediction = "Fraudulent" if fraud_prob >= 0.5 else "Genuine"

    if fraud_prob >= 0.75:
        risk_level = "High"
    elif fraud_prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction":       prediction,
        "fraud_probability": round(fraud_prob, 4),
        "risk_level":       risk_level,
        "reasons":          _build_reasons(job, fraud_prob),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on a DataFrame of job postings."""
    _load_artifacts()

    df_clean = preprocess_dataframe(df.copy())
    X, _     = build_features(df_clean, tfidf=_tfidf, fit=False)
    probs    = _model.predict_proba(X)[:, 1]

    df["fraud_probability"] = probs.round(4)
    df["prediction"]        = ["Fraudulent" if p >= 0.5 else "Genuine" for p in probs]
    df["risk_level"]        = ["High" if p >= 0.75 else ("Medium" if p >= 0.4 else "Low") for p in probs]
    return df


if __name__ == "__main__":
    sample_job = {
        "title": "Work From Home Data Entry – Earn $5000/week",
        "description": "Urgent hiring! No experience needed. Easy income guaranteed. Registration fee required.",
        "company_profile": "",
        "salary_range": "",
        "requirements": "",
        "benefits": "",
        "telecommuting": 1,
        "has_company_logo": 0,
    }
    result = predict_single(sample_job)
    print(result)

