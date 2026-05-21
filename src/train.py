"""
train.py
--------
Loads data, preprocesses, builds features, trains multiple models,
compares them, saves the best one.

Run: python src/train.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocessing import preprocess_dataframe
from feature_engineering import build_features

DATA_PATH   = r"C:\Projects\fraud-job-detection\data\fake_job_postings.csv"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df)} | Fraud: {df['fraudulent'].sum()} ({df['fraudulent'].mean()*100:.1f}%)")
    return df


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "name":      name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"  Precision: {metrics['precision']} | Recall: {metrics['recall']} "
          f"| F1: {metrics['f1']} | ROC-AUC: {metrics['roc_auc']}")
    return model, metrics


def main():
    # 1. Load & clean
    df = load_data()
    df = preprocess_dataframe(df)

    # 2. Build features
    print("\nBuilding features...")
    X, tfidf = build_features(df, fit=True)
    y = df["fraudulent"].values

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Handle imbalance with SMOTE
    print("Applying SMOTE to handle class imbalance...")
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE — Real: {sum(y_train_bal==0)} | Fraud: {sum(y_train_bal==1)}")

    # 5. Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, scale_pos_weight=10, eval_metric="logloss",
                                             use_label_encoder=False, random_state=42),
    }

    # 6. Train & compare
    results = []
    trained_models = {}
    for name, model in models.items():
        trained, metrics = train_and_evaluate(
            name, model, X_train_bal, X_test, y_train_bal, y_test
        )
        results.append(metrics)
        trained_models[name] = trained

    # 7. Model comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(results_df.to_string(index=False))

    # 8. Save best model (highest F1)
    best_name = results_df.iloc[0]["name"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(tfidf,      f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(results_df, f"{MODELS_DIR}/model_comparison.pkl")
    print(f"Saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()

