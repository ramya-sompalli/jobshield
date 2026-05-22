# 🛡️ JobShield — Fraudulent Job Posting Detection System

An end-to-end ML system that detects fake job postings using NLP + Machine Learning.

A machine learning web app that detects fraudulent job postings using NLP and ensemble models.

🔗 **Live App:** [jobshield-5mbvwahuqbqwoumac4gcij.streamlit.app](https://jobshield-5mbvwahuqbqwoumac4gcij.streamlit.app/)

---

## 🎯 Problem

Fake job postings steal personal data, demand money, and mislead job seekers.
This system automatically classifies any job posting as **Genuine** or **Fraudulent** and explains why.

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle (free):
# https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
# Save as: data/fake_job_postings.csv

# 3. Train the model
cd src
python train.py

# 4. Run the dashboard
streamlit run streamlit_app/app.py

# 5. (Optional) Run the API
uvicorn api.main:app --reload
```

---

## 📁 Project Structure

```
fraud-job-detection/
├── data/                      ← Dataset (download from Kaggle)
├── notebooks/01_eda.py        ← Exploratory Data Analysis
├── models/                    ← Saved model + vectorizer (auto-created)
├── src/
│   ├── preprocessing.py       ← Text cleaning + lemmatization
│   ├── feature_engineering.py ← TF-IDF + structured features
│   ├── train.py               ← Train & compare models
│   ├── predict.py             ← Single + batch inference
│   └── explainability.py      ← SHAP explanations
├── api/main.py                ← FastAPI backend
├── streamlit_app/app.py       ← Streamlit UI (5 pages)
├── requirements.txt
├── Dockerfile
└── setup.sh
```

---

## 🧠 ML Pipeline

```
Raw data → Text cleaning → TF-IDF + Feature engineering
         → SMOTE (handle 5% fraud imbalance)
         → Train: Logistic Regression / Random Forest / XGBoost
         → Evaluate: Precision / Recall / F1 / ROC-AUC
         → Best model saved → FastAPI + Streamlit
```

---

## 🔌 API Usage

Start: `uvicorn api.main:app --reload`

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Urgent WFH – Easy $5000/week",
    "description": "No experience needed. Registration fee required.",
    "company_profile": ""
  }'
```

**Response:**
```json
{
  "prediction": "Fraudulent",
  "fraud_probability": 0.87,
  "risk_level": "High",
  "reasons": [
    "Missing company profile",
    "Suspicious keywords: urgent hiring, no experience needed"
  ]
}
```

**Batch CSV:**
```bash
curl -X POST http://localhost:8000/batch-predict \
  -F "file=@jobs.csv"
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🔍 Single Prediction | Paste one job, get fraud score + reasons |
| 📂 Batch Upload | Upload CSV, classify all at once |
| 📊 Analytics | Dataset fraud patterns & charts |
| 🏆 Model Performance | Compare Logistic Reg vs RF vs XGBoost |
| 🧠 Explainability | SHAP feature contributions per prediction |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10 |
| ML | Scikit-learn, XGBoost, imbalanced-learn |
| NLP | NLTK, TF-IDF |
| XAI | SHAP |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Storage | joblib (model serialization) |
| Deploy | Docker |

---

## 📈 Key Metrics (typical on this dataset)

| Model | F1 | ROC-AUC | Recall |
|-------|----|---------|--------|
| Logistic Regression | ~0.72 | ~0.87 | ~0.70 |
| Random Forest | ~0.78 | ~0.91 | ~0.74 |
| **XGBoost** | **~0.82** | **~0.95** | **~0.80** |

> Recall is prioritised over accuracy — missing a fraud is worse than a false alarm.

---

## 🚀 Future Improvements

- [ ] Real-time scraping + prediction pipeline
- [ ] PostgreSQL for production storage
- [ ] Company URL verification via API
- [ ] Fine-tuned BERT embeddings instead of TF-IDF
- [ ] Model drift monitoring with Evidently AI
- [ ] CI/CD with GitHub Actions

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end ML engineering.
Domain: Cybersecurity + HR Tech | Stack: Python · XGBoost · SHAP · FastAPI · Streamlit
