#!/bin/bash
# Insurance Re-Shopping Predictor — Full Pipeline Runner
# Usage: bash scripts/run_pipeline.sh

set -e

echo "============================================"
echo "  Insurance Re-Shopping Predictor Pipeline"
echo "============================================"
echo ""

# Check for dataset
if [ ! -f "data/raw/train.csv" ]; then
    echo "ERROR: data/raw/train.csv not found."
    echo "Download from: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction"
    echo "Place train.csv in data/raw/"
    exit 1
fi

# Step 1: Install dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt --quiet

# Step 2: Run data quality checks
echo ""
echo "[2/4] Running data quality report..."
python src/data_quality.py

# Step 3: Train model
echo ""
echo "[3/4] Training model..."
python src/train.py

# Step 4: Run tests
echo ""
echo "[4/4] Running tests..."
pytest tests/ -v

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Run 'make app' to launch the Streamlit app"
echo "============================================"
