"""
preprocessing.py
----------------
Cleans raw text columns from the job postings dataset.
Steps: lowercase, remove URLs/special chars, stopword removal, lemmatization.
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only runs once)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()             # collapse spaces
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to all relevant text columns.
    Also fills missing values with empty strings.
    """
    TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text)

    return df


if __name__ == "__main__":
    # Quick smoke test
    sample = "URGENT!!! Easy $5000/week from HOME. No experience needed. Visit http://scam.com"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
