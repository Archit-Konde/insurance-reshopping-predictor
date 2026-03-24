"""Data Quality tab component for the Streamlit app."""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_quality import DataQualityReport

# Terminal color palette
COLORS = {
    "bg": "#1e1e1e",
    "surface": "#252526",
    "accent": "#C9A84C",
    "text": "#d4d4d4",
    "muted": "#858585",
    "green": "#28c840",
    "red": "#f44747",
    "blue": "#569cd6",
}


def render_quality_tab():
    """Render the Data Quality Report tab."""
    data_path = os.path.join("data", "raw", "train.csv")

    if not os.path.exists(data_path):
        st.warning("Dataset not found. Place train.csv in data/raw/ to see the quality report.")
        return

    @st.cache_data
    def load_and_analyze():
        df = pd.read_csv(data_path)
        report = DataQualityReport()
        results = report.run(df)
        return df, results, report

    df, results, report = load_and_analyze()

    # Overall quality score
    score = results["quality_score"]
    score_color = COLORS["green"] if score >= 80 else COLORS["accent"] if score >= 60 else COLORS["red"]

    st.markdown(
        f"""
        <div style="text-align: center; padding: 30px; background: {COLORS['surface']};
                    border-radius: 10px; border: 1px solid #3e3e42; margin-bottom: 20px;">
            <div style="font-size: 4rem; font-weight: bold; color: {score_color};
                        font-family: 'JetBrains Mono', monospace;">
                {score}
            </div>
            <div style="font-size: 1rem; color: {COLORS['muted']};">
                Data Quality Score (0-100)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary table
    report_df = report.to_dataframe()
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Class Distribution
    with st.expander("Class Distribution", expanded=True):
        balance = results["class_balance"]
        counts = balance["counts"]
        labels = {0: "Not interested", 1: "Re-shop interested"}

        fig = go.Figure(data=[
            go.Bar(
                x=[labels.get(k, str(k)) for k in sorted(counts.keys())],
                y=[counts[k] for k in sorted(counts.keys())],
                marker_color=[COLORS["muted"], COLORS["accent"]],
                text=[f"{counts[k]:,}<br>({balance['percentages'][k]}%)" for k in sorted(counts.keys())],
                textposition="auto",
            )
        ])
        fig.update_layout(
            title="Target Variable Distribution",
            template="plotly_dark",
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["surface"],
            font=dict(family="JetBrains Mono, monospace", color=COLORS["text"]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Missing Values
    with st.expander("Missing Values"):
        missing = results["missing_values"]
        missing_data = {col: info["count"] for col, info in missing.items()}
        total_missing = sum(missing_data.values())

        if total_missing == 0:
            st.success("No missing values detected across all columns.")
        else:
            fig = px.bar(
                x=list(missing_data.keys()),
                y=list(missing_data.values()),
                labels={"x": "Column", "y": "Missing Count"},
                title="Missing Values per Column",
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor=COLORS["bg"],
                plot_bgcolor=COLORS["surface"],
                font=dict(family="JetBrains Mono, monospace"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Cardinality
    with st.expander("Cardinality per Categorical Feature"):
        cardinality = results["cardinality"]
        card_rows = []
        for col, info in cardinality.items():
            row = {"Column": col, "Unique Values": info["n_unique"]}
            if "values" in info:
                row["Values"] = ", ".join(str(v) for v in info["values"])
            card_rows.append(row)

        if card_rows:
            st.dataframe(pd.DataFrame(card_rows), use_container_width=True, hide_index=True)

    # Annual Premium Distribution
    with st.expander("Annual Premium Distribution"):
        patterns = results["suspicious_patterns"]
        if "premium_outliers" in patterns:
            po = patterns["premium_outliers"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min", f"{po['min_premium']:,.0f}")
            col2.metric("Q1", f"{po['q1']:,.0f}")
            col3.metric("Q3", f"{po['q3']:,.0f}")
            col4.metric("Max", f"{po['max_premium']:,.0f}")

            st.metric("Outliers (IQR method)", f"{po['n_outliers']:,} ({po['outlier_pct']}%)")

    # Sparse Channels
    with st.expander("Sparse Policy Sales Channels"):
        if "sparse_channels" in patterns:
            sc = patterns["sparse_channels"]
            st.write(f"**{sc['n_sparse']}** of {sc['n_total_channels']} channels have fewer than 100 rows ({sc['sparse_pct']}%)")

    # SQL Validation Queries
    with st.expander("SQL-Style Validation Queries"):
        st.code("""
-- Check for missing values
SELECT column_name, COUNT(*) - COUNT(column_name) AS nulls
FROM policies GROUP BY column_name;

-- Check class balance
SELECT Response, COUNT(*) AS n,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM policies GROUP BY Response;

-- Check for duplicate IDs
SELECT COUNT(*) - COUNT(DISTINCT id) AS duplicate_count FROM policies;

-- Flag sparse policy sales channels
SELECT Policy_Sales_Channel, COUNT(*) AS n FROM policies
GROUP BY Policy_Sales_Channel HAVING COUNT(*) < 100;

-- Check annual premium range violations
SELECT COUNT(*) FROM policies WHERE Annual_Premium < 0;

-- Find annual premium outliers (IQR method)
WITH stats AS (
    SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Annual_Premium) AS q1,
           PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Annual_Premium) AS q3
    FROM policies
)
SELECT COUNT(*) FROM policies, stats
WHERE Annual_Premium < q1 - 1.5 * (q3 - q1)
   OR Annual_Premium > q3 + 1.5 * (q3 - q1);
        """, language="sql")
