"""
Training module for Insurance Re-Shopping Predictor.

Uses LightGBM — the industry standard for tabular insurance data:
- Native handling of categorical features
- Fast training with GPU support
- Built-in class weight handling
- SHAP TreeExplainer compatibility
- Regularization via max_depth and min_child_samples

Primary metric: ROC-AUC (robust to class imbalance)
Secondary metric: F1 on positive class (re-shoppers)
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from src.preprocessing import run_preprocessing_pipeline


def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM with GridSearchCV hyperparameter tuning.

    Tunes on validation set using ROC-AUC as the scoring metric.
    """
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [5, 7, -1],
        "learning_rate": [0.05, 0.1],
        "min_child_samples": [20, 50],
        "class_weight": ["balanced", None],
    }

    base_model = LGBMClassifier(
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")

    # Validate on held-out validation set
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"Validation AUC: {val_auc:.4f}")

    return best_model


def evaluate_model(model, X, y, split_name="Test"):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_proba)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    print(f"\n--- {split_name} Set Metrics ---")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"F1 (pos):  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print(f"\nConfusion Matrix ({split_name}):")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

    print(f"\nClassification Report ({split_name}):")
    print(classification_report(y, y_pred, target_names=["Not interested", "Re-shop"]))

    return {
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def get_feature_importance(model, feature_names):
    """Get top feature importances sorted by gain."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    print("\n--- Feature Importance (Top 10 by Gain) ---")
    for _, row in importance_df.head(10).iterrows():
        bar = "#" * int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"  {row['feature']:.<30} {row['importance']:>8.0f}  {bar}")

    return importance_df


def save_model(model, metrics_train, metrics_val, metrics_test, data_info, save_dir="models"):
    """Save model and metadata."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, "lgbm_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Save metadata
    metadata = {
        "train_date": datetime.now().isoformat(),
        "model_type": "LGBMClassifier",
        "val_auc": metrics_val["auc"],
        "test_auc": metrics_test["auc"],
        "train_auc": metrics_train["auc"],
        "val_f1": metrics_val["f1"],
        "test_f1": metrics_test["f1"],
        "val_precision": metrics_val["precision"],
        "test_precision": metrics_test["precision"],
        "val_recall": metrics_val["recall"],
        "test_recall": metrics_test["recall"],
        "n_train": data_info["n_train"],
        "n_val": data_info["n_val"],
        "n_test": data_info["n_test"],
        "positive_rate_train": data_info["positive_rate_train"],
        "positive_rate_test": data_info["positive_rate_test"],
        "best_params": {k: v for k, v in model.get_params().items()
                        if k in ["n_estimators", "max_depth", "learning_rate",
                                 "min_child_samples", "class_weight"]},
    }

    metadata_path = os.path.join(save_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {metadata_path}")

    return metadata


def main():
    """Run the full training pipeline."""
    data_path = os.path.join("data", "raw", "train.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Download from: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction")
        return

    print("=" * 60)
    print("Insurance Re-Shopping Predictor — Training Pipeline")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Preprocess
    print("\n--- Preprocessing ---")
    data = run_preprocessing_pipeline(df)

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    feature_columns = data["feature_columns"]

    data_info = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }

    # Train
    print("\n--- Training ---")
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n--- Evaluation ---")
    metrics_train = evaluate_model(model, X_train, y_train, "Train")
    metrics_val = evaluate_model(model, X_val, y_val, "Validation")
    metrics_test = evaluate_model(model, X_test, y_test, "Test")

    # Feature importance
    get_feature_importance(model, feature_columns)

    # Save
    save_model(model, metrics_train, metrics_val, metrics_test, data_info)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
