"""
Preprocessing pipeline for Insurance Re-Shopping Predictor.

Each step is documented with WHY the decision was made — this is what
"understands the foundational layer of data analysis" means in practice.

Pipeline steps (in exact order):
1. Drop `id` — identifier column, not a feature (leaks row identity)
2. Encode Gender — binary mapping, OHE unnecessary for 2 values
3. Encode Vehicle_Age — ordinal: age has natural order (<1 < 1-2 < >2)
4. Encode Vehicle_Damage — binary mapping (Yes/No → 1/0)
5. Log-transform Annual_Premium — right-skewed, log stabilizes variance
6. StandardScaler on continuous features — LightGBM is scale-invariant
   but scaling helps SHAP interpretation consistency
7. SMOTE on training set only — generates synthetic minority samples
   to address 88/12 class imbalance. CRITICAL: never fit on val/test.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Encoding Maps ─────────────────────────────────────────────────────

GENDER_MAP = {"Female": 0, "Male": 1}

# Ordinal encoding: vehicle age has a natural order
VEHICLE_AGE_MAP = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}

VEHICLE_DAMAGE_MAP = {"No": 0, "Yes": 1}

# Columns to scale — continuous features that benefit from normalization
SCALE_COLUMNS = ["Age", "Annual_Premium_log", "Vintage", "Policy_Sales_Channel"]

# Feature columns after preprocessing (in order)
FEATURE_COLUMNS = [
    "Gender", "Age", "Driving_License", "Region_Code",
    "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
    "Annual_Premium_log", "Policy_Sales_Channel", "Vintage",
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all categorical encodings and transforms.

    Args:
        df: Raw DataFrame with original column names.

    Returns:
        DataFrame with encoded features, id dropped, Annual_Premium
        replaced by Annual_Premium_log.
    """
    df = df.copy()

    # Step 1: Drop id — it's a row identifier, not a predictive feature.
    # Including it would cause the model to memorize row indices.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Step 2: Encode Gender (Female=0, Male=1)
    # Binary feature → simple map. OHE would add a redundant column.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(GENDER_MAP)

    # Step 3: Encode Vehicle_Age as ordinal (0, 1, 2)
    # There's a natural order: newer → older. Ordinal preserves this
    # relationship, which tree models can exploit for splits.
    if "Vehicle_Age" in df.columns:
        df["Vehicle_Age"] = df["Vehicle_Age"].map(VEHICLE_AGE_MAP)

    # Step 4: Encode Vehicle_Damage (No=0, Yes=1)
    # Binary feature, same rationale as Gender.
    if "Vehicle_Damage" in df.columns:
        df["Vehicle_Damage"] = df["Vehicle_Damage"].map(VEHICLE_DAMAGE_MAP)

    # Step 5: Log-transform Annual_Premium
    # The distribution is heavily right-skewed (max ~540K, median ~31K).
    # Log transform compresses the tail and makes the distribution more
    # symmetric, which helps both tree splits and SHAP value stability.
    if "Annual_Premium" in df.columns:
        df["Annual_Premium_log"] = np.log1p(df["Annual_Premium"])
        df = df.drop(columns=["Annual_Premium"])

    return df


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """Apply StandardScaler to continuous features.

    Step 6: StandardScaler on [Age, Annual_Premium_log, Vintage, Policy_Sales_Channel].
    LightGBM is technically scale-invariant (tree splits are ordinal),
    but scaling ensures SHAP values are on a comparable scale for
    interpretation and makes the feature importance more intuitive.

    Args:
        df: Encoded DataFrame.
        scaler: Pre-fitted scaler (for transform-only mode).
        fit: If True, fit a new scaler. If False, use the provided one.

    Returns:
        (scaled_df, scaler) tuple.
    """
    df = df.copy()
    cols_to_scale = [c for c in SCALE_COLUMNS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Apply SMOTE to balance the training set.

    Step 7: SMOTE generates synthetic minority class samples by
    interpolating between existing minority examples. This addresses
    the ~88/12 class imbalance without losing majority class information
    (unlike undersampling).

    CRITICAL: SMOTE is fit ONLY on the training set. Applying it to
    validation or test sets would:
    - Leak information about the minority class distribution
    - Create unrealistic evaluation metrics
    - Defeat the purpose of held-out evaluation
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def split_data(df: pd.DataFrame, target_col: str = "Response", random_state: int = 42):
    """Stratified 80/10/10 train/val/test split.

    Stratification ensures each split preserves the class ratio (~88/12),
    which is critical for:
    - Reliable AUC computation on val/test
    - Consistent positive-class representation across splits
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Second split: 50/50 of temp → 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing_pipeline(df: pd.DataFrame, save_dir: str = "models"):
    """Execute the full preprocessing pipeline and save artifacts.

    Args:
        df: Raw training DataFrame.
        save_dir: Directory to save pipeline artifacts.

    Returns:
        Dictionary with all splits and the fitted scaler.
    """
    print("Step 1: Encoding features...")
    df_encoded = encode_features(df)

    print("Step 2: Splitting data (80/10/10 stratified)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)

    print(f"  Train: {len(X_train):,} rows ({y_train.mean():.2%} positive)")
    print(f"  Val:   {len(X_val):,} rows ({y_val.mean():.2%} positive)")
    print(f"  Test:  {len(X_test):,} rows ({y_test.mean():.2%} positive)")

    print("Step 3: Scaling continuous features...")
    X_train_scaled, scaler = scale_features(X_train, fit=True)
    X_val_scaled, _ = scale_features(X_val, scaler=scaler, fit=False)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler, fit=False)

    print("Step 4: Applying SMOTE to training set only...")
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)
    print(f"  After SMOTE: {len(X_train_resampled):,} rows ({y_train_resampled.mean():.2%} positive)")

    # Save pipeline artifacts
    os.makedirs(save_dir, exist_ok=True)
    pipeline_path = os.path.join(save_dir, "preprocessing_pipeline.pkl")
    joblib.dump({
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "scale_columns": SCALE_COLUMNS,
        "gender_map": GENDER_MAP,
        "vehicle_age_map": VEHICLE_AGE_MAP,
        "vehicle_damage_map": VEHICLE_DAMAGE_MAP,
    }, pipeline_path)
    print(f"\nPipeline saved to {pipeline_path}")

    return {
        "X_train": X_train_resampled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_resampled,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    }


def preprocess_single_input(input_dict: dict, pipeline_path: str = "models/preprocessing_pipeline.pkl"):
    """Preprocess a single user input for prediction.

    Used by the Streamlit app to transform form inputs into model-ready features.

    Args:
        input_dict: Dictionary with raw feature values from the form.
        pipeline_path: Path to the saved preprocessing pipeline.

    Returns:
        DataFrame with one row of preprocessed features.
    """
    pipeline = joblib.load(pipeline_path)
    scaler = pipeline["scaler"]

    df = pd.DataFrame([input_dict])
    df = encode_features(df)

    # Ensure column order matches training
    for col in pipeline["feature_columns"]:
        if col not in df.columns:
            df[col] = 0

    df = df[pipeline["feature_columns"]]
    df, _ = scale_features(df, scaler=scaler, fit=False)

    return df


def main():
    """Run preprocessing pipeline on the training dataset."""
    data_path = os.path.join("data", "raw", "train.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows\n")

    result = run_preprocessing_pipeline(df)
    print(f"\nPreprocessing complete.")
    print(f"Features: {result['feature_columns']}")


if __name__ == "__main__":
    main()
