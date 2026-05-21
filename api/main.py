"""
api/main.py
-----------
FastAPI backend exposing predict, batch-predict, and health endpoints.

Start server: uvicorn api.main:app --reload --port 8000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from predict import predict_single, predict_batch
from explainability import explain_prediction

app = FastAPI(
    title="JobShield — Fraudulent Job Detection API",
    description="Detects fake job postings using ML + NLP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ──────────────────────────────────────────────────────────

class JobPosting(BaseModel):
    title:             str
    description:       str
    company_profile:   Optional[str] = ""
    salary_range:      Optional[str] = ""
    requirements:      Optional[str] = ""
    benefits:          Optional[str] = ""
    telecommuting:     Optional[int] = 0
    has_company_logo:  Optional[int] = 1
    employment_type:   Optional[str] = ""
    industry:          Optional[str] = ""
    location:          Optional[str] = ""


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "JobShield API"}


@app.post("/predict")
def predict(job: JobPosting):
    """
    Predict whether a single job posting is genuine or fraudulent.
    
    Returns fraud probability, risk level, and human-readable reasons.
    """
    try:
        result = predict_single(job.dict())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain")
def predict_with_explanation(job: JobPosting):
    """
    Predict + return SHAP explanation with top contributing features.
    """
    try:
        prediction = predict_single(job.dict())
        explanation = explain_prediction(job.dict())
        return {**prediction, "explanation": explanation["top_features"]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Upload a CSV of job postings and get fraud predictions for all rows.
    CSV must have at minimum: title, description columns.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required = ["title", "description"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    try:
        result_df = predict_batch(df)
        return result_df[["title", "fraud_probability", "prediction", "risk_level"]].to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
