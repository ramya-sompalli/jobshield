"""
streamlit_app/app.py
--------------------
Beautiful 5-page Streamlit dashboard for JobShield.

Run: streamlit run streamlit_app/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import base64
from io import BytesIO

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobShield — Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = r"C:\Projects\fraud-job-detection\models"
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "fake_job_postings.csv")


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_predict():
    from predict import predict_single, predict_batch
    return predict_single, predict_batch

@st.cache_resource
def load_explainer():
    from explainability import explain_prediction
    return explain_prediction

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

def risk_badge(level):
    colors = {"High": "#E24B4A", "Medium": "#EF9F27", "Low": "#639922"}
    c = colors.get(level, "#888")
    return f'<span style="background:{c};color:white;padding:3px 10px;border-radius:12px;font-weight:600">{level}</span>'


# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("JobShield 🛡️")
st.sidebar.caption("Fraudulent Job Posting Detector")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Single Job Prediction", "📂 Batch CSV Upload",
     "📊 Analytics Dashboard", "🏆 Model Performance", "🧠 Explainability (SHAP)"]
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Job Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Job Prediction":
    st.title("🔍 Single Job Prediction")
    st.caption("Paste a job posting and get an instant fraud probability score.")

    col1, col2 = st.columns([2, 1])

    with col1:
        title       = st.text_input("Job Title *", placeholder="e.g. Senior Software Engineer")
        description = st.text_area("Job Description *", height=160,
                                   placeholder="Paste the full job description here...")
        company     = st.text_area("Company Profile", height=80,
                                   placeholder="About the company...")
        col_a, col_b = st.columns(2)
        with col_a:
            salary  = st.text_input("Salary Range", placeholder="e.g. $50,000 – $70,000")
            emptype = st.selectbox("Employment Type",
                                   ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"])
        with col_b:
            location = st.text_input("Location", placeholder="e.g. Mumbai, India")
            industry = st.text_input("Industry", placeholder="e.g. Information Technology")

        col_c, col_d = st.columns(2)
        with col_c:
            telecommute = st.checkbox("Remote / Work From Home")
        with col_d:
            has_logo = st.checkbox("Company has logo", value=True)

        requirements = st.text_area("Requirements", height=60, placeholder="Skills required...")
        benefits     = st.text_area("Benefits", height=60, placeholder="Benefits offered...")

    with col2:
        st.markdown("### Result")
        if st.button("🔎 Analyse Job Posting", use_container_width=True, type="primary"):
            if not title or not description:
                st.error("Please fill in Job Title and Description.")
            else:
                try:
                    predict_single, _ = load_predict()
                    job = {
                        "title": title, "description": description,
                        "company_profile": company, "salary_range": salary,
                        "requirements": requirements, "benefits": benefits,
                        "telecommuting": int(telecommute), "has_company_logo": int(has_logo),
                        "employment_type": emptype, "industry": industry, "location": location,
                    }
                    with st.spinner("Analysing..."):
                        result = predict_single(job)

                    prob = result["fraud_probability"]
                    pred = result["prediction"]
                    risk = result["risk_level"]

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Fraud Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#E24B4A" if prob > 0.5 else "#639922"},
                            "steps": [
                                {"range": [0, 40],   "color": "#EAF3DE"},
                                {"range": [40, 70],  "color": "#FAEEDA"},
                                {"range": [70, 100], "color": "#FCEBEB"},
                            ],
                        }
                    ))
                    fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"**Prediction:** {'🚨 ' if pred=='Fraudulent' else '✅ '}{pred}")
                    st.markdown(f"**Risk Level:** {risk_badge(risk)}", unsafe_allow_html=True)

                    st.markdown("**⚠️ Reasons:**")
                    for r in result["reasons"]:
                        st.markdown(f"- {r}")

                except FileNotFoundError:
                    st.warning("⚠️ Model not trained yet. Run `python src/train.py` first.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.info("💡 Fields marked * are required. More detail = better accuracy.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch CSV Upload
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch CSV Upload":
    st.title("📂 Batch CSV Upload")
    st.caption("Upload a CSV with multiple job postings to classify all at once.")

    st.download_button(
        "⬇️ Download sample CSV template",
        data="title,description,company_profile,salary_range\nSoftware Engineer,Build features...,Tech Corp,60000-80000\n",
        file_name="sample_jobs.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df)} rows detected**")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            try:
                _, predict_batch = load_predict()
                with st.spinner("Classifying all postings..."):
                    result_df = predict_batch(df.copy())

                fraud_count = (result_df["prediction"] == "Fraudulent").sum()
                total       = len(result_df)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Jobs",    total)
                col2.metric("Fraudulent",    fraud_count, delta=f"{fraud_count/total*100:.1f}%", delta_color="inverse")
                col3.metric("Genuine",       total - fraud_count)

                display_cols = ["title", "fraud_probability", "prediction", "risk_level"]
                avail = [c for c in display_cols if c in result_df.columns]
                st.dataframe(result_df[avail].style.background_gradient(
                    subset=["fraud_probability"], cmap="RdYlGn_r"
                ), use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download results CSV", csv_out, "fraud_results.csv", "text/csv")

            except FileNotFoundError:
                st.warning("⚠️ Model not trained yet. Run `python src/train.py` first.")
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.caption("Explore patterns in the dataset.")

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found. Download from Kaggle and place in `data/fake_job_postings.csv`.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Postings",    len(df))
        col2.metric("Fraudulent",        df["fraudulent"].sum())
        col3.metric("Fraud Rate",        f"{df['fraudulent'].mean()*100:.1f}%")
        col4.metric("Genuine",           (df["fraudulent"] == 0).sum())

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(
                values=df["fraudulent"].value_counts().values,
                names=["Genuine", "Fraudulent"],
                color_discrete_sequence=["#639922", "#E24B4A"],
                title="Fraud vs Genuine distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "employment_type" in df.columns:
                emp_fraud = df.groupby("employment_type")["fraudulent"].mean().reset_index()
                emp_fraud.columns = ["Employment Type", "Fraud Rate"]
                emp_fraud = emp_fraud.dropna().sort_values("Fraud Rate", ascending=False)
                fig2 = px.bar(emp_fraud, x="Employment Type", y="Fraud Rate",
                              color="Fraud Rate", color_continuous_scale="RdYlGn_r",
                              title="Fraud rate by employment type")
                st.plotly_chart(fig2, use_container_width=True)

        if "telecommuting" in df.columns:
            tele_fraud = df.groupby("telecommuting")["fraudulent"].mean().reset_index()
            tele_fraud["telecommuting"] = tele_fraud["telecommuting"].map({0: "On-site", 1: "Remote"})
            fig3 = px.bar(tele_fraud, x="telecommuting", y="fraudulent",
                          labels={"telecommuting": "Work Type", "fraudulent": "Fraud Rate"},
                          color="fraudulent", color_continuous_scale="RdYlGn_r",
                          title="Fraud rate: Remote vs On-site")
            st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Performance":
    st.title("🏆 Model Performance")

    comp_path = r"C:\Projects\fraud-job-detection\models\model_comparison.pkl"
    if not os.path.exists(comp_path):
        st.warning("Train the model first: `python src/train.py`")
    else:
        comp_df = joblib.load(comp_path)
        st.dataframe(comp_df.style.highlight_max(
            subset=["f1", "roc_auc", "recall"],
            color="#EAF3DE"
        ), use_container_width=True)

        fig = px.bar(
            comp_df.melt(id_vars="name", value_vars=["precision", "recall", "f1", "roc_auc"]),
            x="name", y="value", color="variable", barmode="group",
            labels={"name": "Model", "value": "Score", "variable": "Metric"},
            title="Model comparison across metrics",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("📌 For fraud detection, **Recall** is the most important metric — "
                "we want to catch as many frauds as possible, even at the cost of some false positives.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Explainability (SHAP)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability (SHAP)":
    st.title("🧠 Explainability — SHAP")
    st.caption("Understand WHY the model flagged a posting as fraudulent.")

    title       = st.text_input("Job Title", value="Urgent Work From Home — Easy $3000/week")
    description = st.text_area("Description",
                                value="No experience needed. Immediate joining. Registration fee required. Easy income guaranteed.",
                                height=100)
    company     = st.text_input("Company Profile", value="")

    if st.button("🔬 Explain Prediction", type="primary"):
        try:
            explain = load_explainer()
            job = {
                "title": title, "description": description,
                "company_profile": company, "salary_range": "",
                "requirements": "", "benefits": "",
                "telecommuting": 1, "has_company_logo": 0,
            }
            with st.spinner("Computing SHAP values..."):
                result = explain(job)

            st.markdown("### Top features driving this prediction")
            feats = result["top_features"]
            feat_df = pd.DataFrame(feats, columns=["Feature", "SHAP Value"])
            feat_df["Impact"] = feat_df["SHAP Value"].apply(
                lambda v: "🔴 Increases fraud score" if v > 0 else "🟢 Decreases fraud score"
            )
            st.dataframe(feat_df, use_container_width=True)

            if result.get("chart_base64"):
                img_bytes = base64.b64decode(result["chart_base64"])
                st.image(img_bytes, caption="SHAP feature contribution chart")

        except FileNotFoundError:
            st.warning("Train the model first: `python src/train.py`")
        except Exception as e:
            st.error(f"Error: {e}")


