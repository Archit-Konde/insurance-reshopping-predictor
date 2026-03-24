"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    encode_features,
    scale_features,
    apply_smote,
    split_data,
    GENDER_MAP,
    VEHICLE_AGE_MAP,
    VEHICLE_DAMAGE_MAP,
    FEATURE_COLUMNS,
)


@pytest.fixture
def sample_df():
    """Create a small valid dataset."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "id": range(1, n + 1),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Age": np.random.randint(20, 70, n).astype(np.int64),
        "Driving_License": np.random.choice([0, 1], n, p=[0.01, 0.99]).astype(np.int64),
        "Region_Code": np.random.choice(range(0, 53), n).astype(float),
        "Previously_Insured": np.random.choice([0, 1], n).astype(np.int64),
        "Vehicle_Age": np.random.choice(["< 1 Year", "1-2 Year", "> 2 Years"], n),
        "Vehicle_Damage": np.random.choice(["Yes", "No"], n),
        "Annual_Premium": np.random.uniform(2000, 100000, n),
        "Policy_Sales_Channel": np.random.choice([26.0, 124.0, 152.0, 160.0], n),
        "Vintage": np.random.randint(10, 300, n).astype(np.int64),
        "Response": np.random.choice([0, 1], n, p=[0.88, 0.12]).astype(np.int64),
    })


class TestEncodeFeatures:
    def test_id_dropped(self, sample_df):
        encoded = encode_features(sample_df)
        assert "id" not in encoded.columns

    def test_gender_encoded(self, sample_df):
        encoded = encode_features(sample_df)
        assert set(encoded["Gender"].unique()).issubset({0, 1})

    def test_vehicle_age_ordinal(self, sample_df):
        encoded = encode_features(sample_df)
        assert set(encoded["Vehicle_Age"].unique()).issubset({0, 1, 2})

    def test_vehicle_damage_encoded(self, sample_df):
        encoded = encode_features(sample_df)
        assert set(encoded["Vehicle_Damage"].unique()).issubset({0, 1})

    def test_annual_premium_log_created(self, sample_df):
        encoded = encode_features(sample_df)
        assert "Annual_Premium_log" in encoded.columns
        assert "Annual_Premium" not in encoded.columns

    def test_log_transform_values(self, sample_df):
        encoded = encode_features(sample_df)
        # log1p(x) should be > 0 for all positive premiums
        assert (encoded["Annual_Premium_log"] > 0).all()

    def test_no_mutation_of_input(self, sample_df):
        original_cols = list(sample_df.columns)
        encode_features(sample_df)
        assert list(sample_df.columns) == original_cols


class TestScaleFeatures:
    def test_scaler_returns_tuple(self, sample_df):
        encoded = encode_features(sample_df)
        result = scale_features(encoded, fit=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scaled_mean_near_zero(self, sample_df):
        encoded = encode_features(sample_df)
        scaled_df, scaler = scale_features(encoded, fit=True)
        # Age should be roughly centered around 0 after scaling
        assert abs(scaled_df["Age"].mean()) < 0.1

    def test_transform_mode(self, sample_df):
        encoded = encode_features(sample_df)
        _, scaler = scale_features(encoded, fit=True)
        # Apply to same data in transform mode
        scaled_df, _ = scale_features(encoded, scaler=scaler, fit=False)
        assert abs(scaled_df["Age"].mean()) < 0.1


class TestSMOTE:
    def test_smote_balances_classes(self, sample_df):
        encoded = encode_features(sample_df)
        X = encoded.drop(columns=["Response"])
        y = encoded["Response"]
        X_res, y_res = apply_smote(X, y)
        # After SMOTE, classes should be balanced
        counts = y_res.value_counts()
        assert counts[0] == counts[1]

    def test_smote_increases_minority(self, sample_df):
        encoded = encode_features(sample_df)
        X = encoded.drop(columns=["Response"])
        y = encoded["Response"]
        original_minority = (y == 1).sum()
        _, y_res = apply_smote(X, y)
        new_minority = (y_res == 1).sum()
        assert new_minority > original_minority


class TestSplitData:
    def test_split_sizes(self, sample_df):
        encoded = encode_features(sample_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(encoded)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(encoded)
        # Approximate 80/10/10 split
        assert abs(len(X_train) / total - 0.8) < 0.05
        assert abs(len(X_val) / total - 0.1) < 0.05
        assert abs(len(X_test) / total - 0.1) < 0.05

    def test_stratification_preserved(self, sample_df):
        encoded = encode_features(sample_df)
        _, _, _, y_train, y_val, y_test = split_data(encoded)
        # Positive rates should be similar across splits
        rates = [y_train.mean(), y_val.mean(), y_test.mean()]
        assert max(rates) - min(rates) < 0.1  # within 10pp

    def test_no_data_leakage(self, sample_df):
        encoded = encode_features(sample_df)
        X_train, X_val, X_test, _, _, _ = split_data(encoded)
        # No overlapping indices
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
