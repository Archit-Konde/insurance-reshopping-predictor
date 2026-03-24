"""Tests for the DataQualityReport class."""

import numpy as np
import pandas as pd
import pytest

from src.data_quality import DataQualityReport  # noqa: F401 EXPECTED_COLUMNS used indirectly via schema


@pytest.fixture
def sample_df():
    """Create a small valid dataset matching the expected schema."""
    np.random.seed(42)
    n = 200
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


@pytest.fixture
def report():
    return DataQualityReport()


class TestSchemaCheck:
    def test_valid_schema_passes(self, report, sample_df):
        results = report.run(sample_df)
        assert results["schema_check"]["pass"] is True
        assert results["schema_check"]["columns_present"] is True
        assert results["schema_check"]["missing_columns"] == []

    def test_missing_column_detected(self, report, sample_df):
        df = sample_df.drop(columns=["Age"])
        results = report.run(df)
        assert results["schema_check"]["columns_present"] is False
        assert "Age" in results["schema_check"]["missing_columns"]

    def test_extra_columns_detected(self, report, sample_df):
        df = sample_df.copy()
        df["extra_col"] = 1
        results = report.run(df)
        assert "extra_col" in results["schema_check"]["extra_columns"]


class TestMissingValues:
    def test_no_missing_values(self, report, sample_df):
        results = report.run(sample_df)
        for col, info in results["missing_values"].items():
            assert info["count"] == 0

    def test_missing_values_detected(self, report, sample_df):
        df = sample_df.copy()
        df.loc[0:4, "Age"] = np.nan
        results = report.run(df)
        assert results["missing_values"]["Age"]["count"] == 5


class TestClassBalance:
    def test_two_classes_present(self, report, sample_df):
        results = report.run(sample_df)
        assert results["class_balance"]["n_classes"] == 2

    def test_imbalance_ratio_computed(self, report, sample_df):
        results = report.run(sample_df)
        ratio = results["class_balance"]["imbalance_ratio"]
        assert 0 < ratio < 1


class TestDuplicateIds:
    def test_no_duplicates(self, report, sample_df):
        results = report.run(sample_df)
        assert results["duplicate_ids"]["duplicate_count"] == 0
        assert results["duplicate_ids"]["has_duplicates"] is False

    def test_duplicates_detected(self, report, sample_df):
        df = sample_df.copy()
        df.loc[0, "id"] = df.loc[1, "id"]
        results = report.run(df)
        assert results["duplicate_ids"]["duplicate_count"] == 1
        assert results["duplicate_ids"]["has_duplicates"] is True


class TestRangeViolations:
    def test_no_violations_on_clean_data(self, report, sample_df):
        results = report.run(sample_df)
        assert results["range_violations"]["Age"]["total"] == 0

    def test_age_violations_detected(self, report, sample_df):
        df = sample_df.copy()
        df.loc[0, "Age"] = 15
        df.loc[1, "Age"] = 90
        results = report.run(df)
        assert results["range_violations"]["Age"]["below_18"] == 1
        assert results["range_violations"]["Age"]["above_85"] == 1
        assert results["range_violations"]["Age"]["total"] == 2


class TestSuspiciousPatterns:
    def test_premium_outliers_detected(self, report, sample_df):
        df = sample_df.copy()
        df.loc[0, "Annual_Premium"] = 1_000_000  # extreme outlier
        results = report.run(df)
        assert results["suspicious_patterns"]["premium_outliers"]["n_outliers"] >= 1


class TestQualityScore:
    def test_score_between_0_and_100(self, report, sample_df):
        results = report.run(sample_df)
        assert 0 <= results["quality_score"] <= 100

    def test_clean_data_scores_high(self, report, sample_df):
        results = report.run(sample_df)
        assert results["quality_score"] >= 70


class TestOutputFormats:
    def test_to_dataframe(self, report, sample_df):
        report.run(sample_df)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 6
        assert "check" in df.columns
        assert "status" in df.columns

    def test_to_markdown(self, report, sample_df):
        report.run(sample_df)
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "Quality Score" in md
        assert "Schema Validation" in md
