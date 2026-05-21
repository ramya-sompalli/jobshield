"""
explainability.py
-----------------
Uses SHAP to explain why a job posting was flagged as fraud.
Returns top contributing features and a base64-encoded chart.
"""

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, os

from preprocessing import preprocess_dataframe
from feature_engineering import build_features

MODELS_DIR = r"C:\Projects\fraud-job-detection\models"

_model     = None
_tfidf     = None
_explainer = None


def _load():
    global _model, _tfidf, _explainer
    if _model is None:
        _model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        _tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        # TreeExplainer works for XGBoost / RandomForest
        # LinearExplainer for LogisticRegression
        try:
            _explainer = shap.TreeExplainer(_model)
        except Exception:
            _explainer = shap.LinearExplainer(_model, shap.maskers.Independent(
                np.zeros((1, _model.n_features_in_))
            ))


def explain_prediction(job: dict, top_n: int = 10) -> dict:
    """
    Returns SHAP values for the top N features + a base64 bar chart.
    """
    _load()

    df = pd.DataFrame([job])
    df = preprocess_dataframe(df)
    X, _ = build_features(df, tfidf=_tfidf, fit=False)

    shap_values = _explainer.shap_values(X)

    # For binary classification some models return list of 2 arrays
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Feature names = TF-IDF vocab + structured feature names
    tfidf_features = _tfidf.get_feature_names_out().tolist()
    struct_features = [
        "desc_length", "title_length", "profile_length",
        "missing_salary", "missing_company_profile", "missing_requirements",
        "is_remote", "has_logo", "suspicious_keyword_count"
    ]
    feature_names = tfidf_features + struct_features

    # Top N by absolute SHAP value
    idx = np.argsort(np.abs(sv))[::-1][:top_n]
    top_features = [(feature_names[i], float(sv[i])) for i in idx]

    # Generate bar chart
    names  = [f[0] for f in top_features][::-1]
    values = [f[1] for f in top_features][::-1]
    colors = ["#E24B4A" if v > 0 else "#378ADD" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SHAP value  (positive = more fraudulent)")
    ax.set_title("Feature contributions to fraud prediction")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    chart_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "top_features": top_features,
        "chart_base64": chart_b64,
    }


if __name__ == "__main__":
    print("Explainability module ready. Run after training.")




