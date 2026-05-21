#!/bin/bash
# setup.sh — One-time setup for JobShield project
# Run: bash setup.sh

echo "======================================"
echo "  JobShield — Project Setup"
echo "======================================"

# 1. Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# 2. Download NLTK data
echo ""
echo "[2/4] Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# 3. Dataset instructions
echo ""
echo "[3/4] Dataset setup..."
echo "---------------------------------------"
echo "  Download the dataset manually from Kaggle:"
echo "  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction"
echo "  Save file as:  data/fake_job_postings.csv"
echo "---------------------------------------"

# 4. Done
echo ""
echo "[4/4] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Place fake_job_postings.csv in data/"
echo "  2. Train model:   cd src && python train.py"
echo "  3. Run dashboard: streamlit run streamlit_app/app.py"
echo "  4. Run API:       uvicorn api.main:app --reload"
echo ""
