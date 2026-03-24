# Data Directory

## Dataset: Health Insurance Cross Sell Prediction

**Source**: [Kaggle — Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

**Size**: 381,109 rows x 12 columns

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Unique identifier for each customer |
| `Gender` | object | Male / Female |
| `Age` | int64 | Age of the customer (20–85) |
| `Driving_License` | int64 | 0 = no license, 1 = has license |
| `Region_Code` | float64 | Geographic region code (0–52) |
| `Previously_Insured` | int64 | 0 = not previously insured, 1 = already has vehicle insurance |
| `Vehicle_Age` | object | "< 1 Year", "1-2 Year", "> 2 Years" |
| `Vehicle_Damage` | object | "Yes" = vehicle was damaged before, "No" = no prior damage |
| `Annual_Premium` | float64 | Annual premium amount (INR) |
| `Policy_Sales_Channel` | float64 | Anonymized channel code (1–163) |
| `Vintage` | int64 | Number of days the customer has been associated (10–299) |
| `Response` | int64 | **Target** — 1 = interested in vehicle insurance, 0 = not interested |

## Class Imbalance

The target variable is heavily imbalanced:
- **Response = 0** (not interested): ~87.74%
- **Response = 1** (interested / re-shop): ~12.26%

This 88/12 split requires careful handling:
- SMOTE applied to training set only (never validation/test)
- ROC-AUC used as primary metric (robust to imbalance)
- Stratified splitting preserves class ratio across all splits

## Encoding Conventions

| Feature | Encoding | Rationale |
|---------|----------|-----------|
| Gender | Female=0, Male=1 | Binary — OHE adds redundant column |
| Vehicle_Age | <1 Year=0, 1-2 Year=1, >2 Years=2 | Ordinal — natural age order |
| Vehicle_Damage | No=0, Yes=1 | Binary |
| Annual_Premium | log1p transform | Right-skewed distribution |
| Continuous features | StandardScaler | Consistent SHAP interpretation |

## Setup

1. Download `train.csv` from the Kaggle link above
2. Place it in `data/raw/train.csv`
3. Run `make quality` to generate the data quality report
4. Run `make train` to preprocess and train the model
