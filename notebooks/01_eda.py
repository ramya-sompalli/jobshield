"""
notebooks/01_eda.py
-------------------
Exploratory Data Analysis for the Fake Job Postings dataset.
Run cell by cell in Jupyter or as a plain script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/fake_job_postings.csv")
print("Shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nFirst 3 rows:\n", df.head(3))

# ── 2. Class distribution ────────────────────────────────────────────────────
print("\n--- Class Distribution ---")
print(df["fraudulent"].value_counts())
print(f"Fraud rate: {df['fraudulent'].mean()*100:.2f}%")

fig, ax = plt.subplots()
df["fraudulent"].value_counts().plot.pie(
    labels=["Genuine", "Fraudulent"],
    colors=["#639922", "#E24B4A"],
    autopct="%1.1f%%", ax=ax
)
ax.set_title("Fraud vs Genuine")
plt.savefig("notebooks/01_class_dist.png", dpi=100, bbox_inches="tight")
plt.close()

# ── 3. Missing values ────────────────────────────────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# ── 4. Word cloud for fraudulent jobs ────────────────────────────────────────
fraud_text = " ".join(df[df["fraudulent"] == 1]["description"].dropna())
wc = WordCloud(width=800, height=400, background_color="white",
               colormap="Reds", max_words=100).generate(fraud_text)
plt.figure(figsize=(10, 4))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most common words in FRAUDULENT job postings")
plt.savefig("notebooks/02_wordcloud_fraud.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved word clouds.")

# ── 5. Fraud by employment type ──────────────────────────────────────────────
if "employment_type" in df.columns:
    et = df.groupby("employment_type")["fraudulent"].mean().sort_values(ascending=False)
    print("\n--- Fraud rate by employment type ---")
    print(et)

# ── 6. Remote vs on-site fraud rate ─────────────────────────────────────────
if "telecommuting" in df.columns:
    print("\n--- Fraud rate: remote vs on-site ---")
    print(df.groupby("telecommuting")["fraudulent"].mean())

print("\nEDA complete. Charts saved to notebooks/")
