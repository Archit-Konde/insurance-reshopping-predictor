"""
Data Quality Report for Insurance Re-Shopping Predictor.

Treats data quality as the primary engineering concern. Every check is
documented with equivalent SQL-style validation queries.

SQL-style validation queries (for reviewers who think in SQL):

    -- Check for missing values
    SELECT column_name, COUNT(*) - COUNT(column_name) AS nulls
    FROM policies GROUP BY column_name;

    -- Check class balance
    SELECT Response, COUNT(*) AS n, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
    FROM policies GROUP BY Response;

    -- Check for duplicate IDs
    SELECT COUNT(*) - COUNT(DISTINCT id) AS duplicate_count FROM policies;

    -- Check region code validity
    SELECT Region_Code, COUNT(*) FROM policies
    GROUP BY Region_Code HAVING Region_Code NOT BETWEEN 0 AND 52;

    -- Flag sparse policy sales channels (<100 rows)
    SELECT Policy_Sales_Channel, COUNT(*) AS n FROM policies
    GROUP BY Policy_Sales_Channel HAVING COUNT(*) < 100;

    -- Check annual premium range violations
    SELECT COUNT(*) FROM policies WHERE Annual_Premium < 0;

    -- Check age range violations
    SELECT COUNT(*) FROM policies WHERE Age < 18 OR Age > 85;

    -- Find annual premium outliers (IQR method)
    WITH stats AS (
        SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Annual_Premium) AS q1,
               PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Annual_Premium) AS q3
        FROM policies
    )
    SELECT COUNT(*) FROM policies, stats
    WHERE Annual_Premium < q1 - 1.5 * (q3 - q1) OR Annual_Premium > q3 + 1.5 * (q3 - q1);
"""

import os
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "id", "Gender", "Age", "Driving_License", "Region_Code",
    "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
]

EXPECTED_DTYPES = {
    "id": "int64",
    "Gender": "object",
    "Age": "int64",
    "Driving_License": "int64",
    "Region_Code": "float64",
    "Previously_Insured": "int64",
    "Vehicle_Age": "object",
    "Vehicle_Damage": "object",
    "Annual_Premium": "float64",
    "Policy_Sales_Channel": "float64",
    "Vintage": "int64",
    "Response": "int64",
}

CATEGORICAL_COLUMNS = ["Gender", "Vehicle_Age", "Vehicle_Damage"]


class DataQualityReport:
    """Run comprehensive data quality checks on the insurance dataset.

    Produces a structured report covering schema validation, missing values,
    class balance, cardinality, range violations, duplicate detection,
    suspicious patterns, and an overall quality score.
    """

    def __init__(self):
        self.results = {}

    def run(self, df: pd.DataFrame) -> dict:
        """Execute all quality checks and return structured results."""
        self.results = {
            "schema_check": self._check_schema(df),
            "missing_values": self._check_missing(df),
            "class_balance": self._check_class_balance(df),
            "cardinality": self._check_cardinality(df),
            "range_violations": self._check_range_violations(df),
            "duplicate_ids": self._check_duplicates(df),
            "suspicious_patterns": self._check_suspicious_patterns(df),
            "quality_score": 0,  # computed after all checks
        }
        self.results["quality_score"] = self._compute_quality_score()
        return self.results

    # ── Schema Check ──────────────────────────────────────────────────

    def _check_schema(self, df: pd.DataFrame) -> dict:
        """Validate all expected columns are present with correct dtypes."""
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

        dtype_mismatches = {}
        for col, expected_dtype in EXPECTED_DTYPES.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                if actual != expected_dtype:
                    dtype_mismatches[col] = {
                        "expected": expected_dtype,
                        "actual": actual,
                    }

        return {
            "columns_present": len(missing_cols) == 0,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "dtype_mismatches": dtype_mismatches,
            "n_columns": len(df.columns),
            "n_rows": len(df),
            "pass": len(missing_cols) == 0 and len(dtype_mismatches) == 0,
        }

    # ── Missing Values ────────────────────────────────────────────────

    def _check_missing(self, df: pd.DataFrame) -> dict:
        """Count and percentage of missing values per column."""
        # SQL: SELECT column_name, COUNT(*) - COUNT(column_name) AS nulls FROM policies
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(4)

        return {
            col: {"count": int(missing_count[col]), "pct": float(missing_pct[col])}
            for col in df.columns
        }

    # ── Class Balance ─────────────────────────────────────────────────

    def _check_class_balance(self, df: pd.DataFrame) -> dict:
        """Target distribution — expect ~88/12 split."""
        # SQL: SELECT Response, COUNT(*), ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2)
        if "Response" not in df.columns:
            return {"error": "Response column not found"}

        counts = df["Response"].value_counts()
        total = len(df)

        return {
            "counts": {int(k): int(v) for k, v in counts.items()},
            "percentages": {int(k): round(v / total * 100, 2) for k, v in counts.items()},
            "imbalance_ratio": round(counts.min() / counts.max(), 4) if len(counts) == 2 else None,
            "n_classes": len(counts),
        }

    # ── Cardinality ───────────────────────────────────────────────────

    def _check_cardinality(self, df: pd.DataFrame) -> dict:
        """Unique value counts per categorical column."""
        result = {}
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                result[col] = {
                    "n_unique": int(df[col].nunique()),
                    "values": df[col].unique().tolist(),
                }
        # Also check high-cardinality columns
        for col in ["Region_Code", "Policy_Sales_Channel"]:
            if col in df.columns:
                result[col] = {
                    "n_unique": int(df[col].nunique()),
                    "top_5": df[col].value_counts().head(5).to_dict(),
                }
        return result

    # ── Range Violations ──────────────────────────────────────────────

    def _check_range_violations(self, df: pd.DataFrame) -> dict:
        """Check Age outside 18–85, Annual_Premium < 0, etc."""
        # SQL: SELECT COUNT(*) FROM policies WHERE Age < 18 OR Age > 85
        violations = {}

        if "Age" in df.columns:
            age_low = int((df["Age"] < 18).sum())
            age_high = int((df["Age"] > 85).sum())
            violations["Age"] = {
                "below_18": age_low,
                "above_85": age_high,
                "total": age_low + age_high,
            }

        if "Annual_Premium" in df.columns:
            # SQL: SELECT COUNT(*) FROM policies WHERE Annual_Premium < 0
            negative = int((df["Annual_Premium"] < 0).sum())
            violations["Annual_Premium_negative"] = negative

        if "Vintage" in df.columns:
            vintage_neg = int((df["Vintage"] < 0).sum())
            violations["Vintage_negative"] = vintage_neg

        if "Driving_License" in df.columns:
            invalid_dl = int(~df["Driving_License"].isin([0, 1]).sum() if df["Driving_License"].isin([0, 1]).all() else (~df["Driving_License"].isin([0, 1])).sum())
            violations["Driving_License_invalid"] = invalid_dl

        return violations

    # ── Duplicate IDs ─────────────────────────────────────────────────

    def _check_duplicates(self, df: pd.DataFrame) -> dict:
        """Count duplicate id values."""
        # SQL: SELECT COUNT(*) - COUNT(DISTINCT id) AS duplicate_count FROM policies
        if "id" not in df.columns:
            return {"error": "id column not found"}

        n_total = len(df)
        n_unique = df["id"].nunique()
        n_duplicates = n_total - n_unique

        return {
            "total_rows": n_total,
            "unique_ids": n_unique,
            "duplicate_count": n_duplicates,
            "has_duplicates": n_duplicates > 0,
        }

    # ── Suspicious Patterns ───────────────────────────────────────────

    def _check_suspicious_patterns(self, df: pd.DataFrame) -> dict:
        """Flag region codes outside expected range, sparse channels, premium outliers."""
        patterns = {}

        # Region_Code outside 0–52
        # SQL: SELECT Region_Code, COUNT(*) FROM policies
        #      GROUP BY Region_Code HAVING Region_Code NOT BETWEEN 0 AND 52
        if "Region_Code" in df.columns:
            invalid_regions = df[~df["Region_Code"].between(0, 52)]
            patterns["invalid_region_codes"] = {
                "count": len(invalid_regions),
                "values": sorted(invalid_regions["Region_Code"].unique().tolist()) if len(invalid_regions) > 0 else [],
            }

        # Sparse policy sales channels (<100 rows)
        # SQL: SELECT Policy_Sales_Channel, COUNT(*) FROM policies
        #      GROUP BY Policy_Sales_Channel HAVING COUNT(*) < 100
        if "Policy_Sales_Channel" in df.columns:
            channel_counts = df["Policy_Sales_Channel"].value_counts()
            sparse = channel_counts[channel_counts < 100]
            patterns["sparse_channels"] = {
                "n_sparse": len(sparse),
                "n_total_channels": len(channel_counts),
                "sparse_pct": round(len(sparse) / len(channel_counts) * 100, 1),
                "examples": sparse.head(10).to_dict(),
            }

        # Annual_Premium outliers (IQR method)
        # SQL: WITH stats AS (SELECT PERCENTILE_CONT(0.25)..., PERCENTILE_CONT(0.75)...)
        if "Annual_Premium" in df.columns:
            q1 = df["Annual_Premium"].quantile(0.25)
            q3 = df["Annual_Premium"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[
                (df["Annual_Premium"] < lower_bound) | (df["Annual_Premium"] > upper_bound)
            ]
            patterns["premium_outliers"] = {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "n_outliers": len(outliers),
                "outlier_pct": round(len(outliers) / len(df) * 100, 2),
                "max_premium": float(df["Annual_Premium"].max()),
                "min_premium": float(df["Annual_Premium"].min()),
            }

        return patterns

    # ── Quality Score ─────────────────────────────────────────────────

    def _compute_quality_score(self) -> float:
        """Weighted 0–100 score across all checks.

        Weights:
            schema_check:        25  (foundational — wrong schema = everything breaks)
            missing_values:      20  (completeness is critical for ML)
            class_balance:       10  (expected imbalance, so lower weight)
            duplicate_ids:       15  (duplicates bias model evaluation)
            range_violations:    15  (out-of-range data = bad data)
            suspicious_patterns: 15  (domain-specific sanity checks)
        """
        score = 0.0

        # Schema (25 pts)
        schema = self.results.get("schema_check", {})
        if schema.get("pass", False):
            score += 25.0
        elif schema.get("columns_present", False):
            score += 15.0  # columns ok but dtype issues

        # Missing values (20 pts) — deduct proportionally
        missing = self.results.get("missing_values", {})
        if missing:
            max_pct = max((v["pct"] for v in missing.values()), default=0)
            if max_pct == 0:
                score += 20.0
            else:
                score += max(0, 20.0 * (1 - max_pct / 100))

        # Class balance (10 pts)
        balance = self.results.get("class_balance", {})
        if balance.get("n_classes") == 2:
            ratio = balance.get("imbalance_ratio", 0)
            # Expected ~0.14 ratio; penalize only if <0.05 or >0.5
            if 0.05 <= ratio <= 0.5:
                score += 10.0
            elif ratio > 0:
                score += 5.0

        # Duplicates (15 pts)
        dupes = self.results.get("duplicate_ids", {})
        if dupes.get("duplicate_count", 1) == 0:
            score += 15.0
        else:
            dupe_pct = dupes.get("duplicate_count", 0) / max(dupes.get("total_rows", 1), 1) * 100
            score += max(0, 15.0 * (1 - dupe_pct / 10))

        # Range violations (15 pts)
        violations = self.results.get("range_violations", {})
        total_violations = 0
        for key, val in violations.items():
            if isinstance(val, dict):
                total_violations += val.get("total", 0)
            else:
                total_violations += val
        if total_violations == 0:
            score += 15.0
        else:
            n_rows = self.results.get("schema_check", {}).get("n_rows", 1)
            violation_pct = total_violations / max(n_rows, 1) * 100
            score += max(0, 15.0 * (1 - violation_pct / 5))

        # Suspicious patterns (15 pts)
        patterns = self.results.get("suspicious_patterns", {})
        pattern_deductions = 0
        if patterns.get("invalid_region_codes", {}).get("count", 0) > 0:
            pattern_deductions += 5
        if patterns.get("premium_outliers", {}).get("outlier_pct", 0) > 5:
            pattern_deductions += 5
        sparse_pct = patterns.get("sparse_channels", {}).get("sparse_pct", 0)
        if sparse_pct > 50:
            pattern_deductions += 5
        score += max(0, 15.0 - pattern_deductions)

        return round(min(100.0, max(0.0, score)), 1)

    # ── Output Formatters ─────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame with one row per check."""
        rows = []

        # Schema
        schema = self.results.get("schema_check", {})
        rows.append({
            "check": "Schema Validation",
            "status": "PASS" if schema.get("pass") else "FAIL",
            "details": f"{schema.get('n_columns', 0)} cols, {schema.get('n_rows', 0)} rows",
            "severity": "critical",
        })

        # Missing values
        missing = self.results.get("missing_values", {})
        total_missing = sum(v["count"] for v in missing.values())
        rows.append({
            "check": "Missing Values",
            "status": "PASS" if total_missing == 0 else "WARN",
            "details": f"{total_missing} total missing values",
            "severity": "high" if total_missing > 0 else "ok",
        })

        # Class balance
        balance = self.results.get("class_balance", {})
        pcts = balance.get("percentages", {})
        rows.append({
            "check": "Class Balance",
            "status": "WARN" if balance.get("imbalance_ratio", 1) < 0.2 else "PASS",
            "details": f"Positive rate: {pcts.get(1, 'N/A')}%",
            "severity": "medium",
        })

        # Duplicates
        dupes = self.results.get("duplicate_ids", {})
        rows.append({
            "check": "Duplicate IDs",
            "status": "PASS" if dupes.get("duplicate_count", 1) == 0 else "FAIL",
            "details": f"{dupes.get('duplicate_count', 'N/A')} duplicates",
            "severity": "high" if dupes.get("has_duplicates") else "ok",
        })

        # Range violations
        violations = self.results.get("range_violations", {})
        total_v = sum(
            v.get("total", v) if isinstance(v, dict) else v
            for v in violations.values()
        )
        rows.append({
            "check": "Range Violations",
            "status": "PASS" if total_v == 0 else "WARN",
            "details": f"{total_v} violations found",
            "severity": "medium" if total_v > 0 else "ok",
        })

        # Suspicious patterns
        patterns = self.results.get("suspicious_patterns", {})
        n_outliers = patterns.get("premium_outliers", {}).get("n_outliers", 0)
        n_sparse = patterns.get("sparse_channels", {}).get("n_sparse", 0)
        rows.append({
            "check": "Suspicious Patterns",
            "status": "WARN" if n_outliers > 0 or n_sparse > 0 else "PASS",
            "details": f"{n_outliers} premium outliers, {n_sparse} sparse channels",
            "severity": "medium",
        })

        # Quality score
        rows.append({
            "check": "Overall Quality Score",
            "status": str(self.results.get("quality_score", 0)),
            "details": "Weighted score (0-100)",
            "severity": "summary",
        })

        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Formatted markdown string for README inclusion."""
        lines = ["## Data Quality Report\n"]

        score = self.results.get("quality_score", 0)
        lines.append(f"**Overall Quality Score: {score}/100**\n")

        # Schema
        schema = self.results.get("schema_check", {})
        status = "PASS" if schema.get("pass") else "FAIL"
        lines.append(f"### Schema Validation: {status}")
        lines.append(f"- Columns: {schema.get('n_columns', 'N/A')}")
        lines.append(f"- Rows: {schema.get('n_rows', 'N/A'):,}")
        if schema.get("missing_columns"):
            lines.append(f"- Missing columns: {schema['missing_columns']}")
        lines.append("")

        # Class balance
        balance = self.results.get("class_balance", {})
        pcts = balance.get("percentages", {})
        lines.append("### Class Distribution")
        for label, pct in sorted(pcts.items()):
            name = "Not interested" if label == 0 else "Interested (re-shop)"
            lines.append(f"- {name}: {pct}%")
        lines.append(f"- Imbalance ratio: {balance.get('imbalance_ratio', 'N/A')}")
        lines.append("")

        # Missing values
        lines.append("### Missing Values")
        missing = self.results.get("missing_values", {})
        has_missing = any(v["count"] > 0 for v in missing.values())
        if not has_missing:
            lines.append("- No missing values detected.")
        else:
            for col, info in missing.items():
                if info["count"] > 0:
                    lines.append(f"- {col}: {info['count']} ({info['pct']}%)")
        lines.append("")

        # Duplicates
        dupes = self.results.get("duplicate_ids", {})
        lines.append("### Duplicate IDs")
        lines.append(f"- Duplicate count: {dupes.get('duplicate_count', 'N/A')}")
        lines.append("")

        # Suspicious patterns
        patterns = self.results.get("suspicious_patterns", {})
        lines.append("### Suspicious Patterns")
        if patterns.get("premium_outliers"):
            po = patterns["premium_outliers"]
            lines.append(f"- Annual Premium outliers (IQR): {po['n_outliers']} ({po['outlier_pct']}%)")
        if patterns.get("sparse_channels"):
            sc = patterns["sparse_channels"]
            lines.append(f"- Sparse channels (<100 rows): {sc['n_sparse']}/{sc['n_total_channels']}")
        lines.append("")

        return "\n".join(lines)


def main():
    """Run data quality report on the training dataset."""
    data_path = os.path.join("data", "raw", "train.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Download from: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction")
        print("Place train.csv in data/raw/")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns\n")

    report = DataQualityReport()
    results = report.run(df)

    print(report.to_markdown())
    print("\n--- Quality Score ---")
    print(f"Score: {results['quality_score']}/100\n")

    # Save report DataFrame
    report_df = report.to_dataframe()
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    report_df.to_csv(os.path.join("data", "processed", "quality_report.csv"), index=False)
    print("Report saved to data/processed/quality_report.csv")


if __name__ == "__main__":
    main()
