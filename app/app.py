"""
Insurance Re-Shopping Predictor — Streamlit App.

Two tabs:
1. Re-Shopping Predictor: Input profile → probability + SHAP explanation
2. Data Quality Report: Comprehensive quality analysis of the training data
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import streamlit as st

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.components.input_form import render_input_form  # noqa: E402
from app.components.results_panel import (  # noqa: E402
    render_counterfactual,
    render_probability_gauge,
    render_top_factors,
)
from app.components.quality_tab import render_quality_tab  # noqa: E402
from src.explain import get_counterfactual, get_shap_values, get_top_factors, get_waterfall_figure  # noqa: E402
from src.preprocessing import preprocess_single_input  # noqa: E402


st.set_page_config(
    page_title="Insurance Re-Shopping Predictor",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for terminal theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {
        font-family: 'JetBrains Mono', monospace;
    }

    .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Style the tab labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="color: #C9A84C; margin-bottom: 5px;">
            $ insurance-reshopping-predictor
        </h1>
        <p style="color: #858585; font-size: 0.95rem;">
            Predicts whether you'd benefit from re-shopping your car insurance
            — using ML trained on 381K real insurance profiles.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the trained model and preprocessing pipeline."""
    model_path = os.path.join("models", "lgbm_model.pkl")
    pipeline_path = os.path.join("models", "preprocessing_pipeline.pkl")

    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        return None, None

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline


# Tabs
tab1, tab2 = st.tabs(["Re-Shopping Predictor", "Data Quality Report"])

with tab1:
    model, pipeline = load_model()

    if model is None:
        st.error(
            "Model not found. Run `make train` to train the model first.\n\n"
            "```bash\n"
            "# 1. Place train.csv in data/raw/\n"
            "# 2. Run the training pipeline:\n"
            "make all\n"
            "```"
        )
    else:
        col_input, col_results = st.columns([4, 6])

        with col_input:
            input_data = render_input_form()

        # Store prediction results in session state to prevent jitter on rerender
        if input_data is not None:
            input_df = preprocess_single_input(input_data, pipeline=pipeline)
            feature_names = pipeline["feature_columns"]

            probability = float(model.predict_proba(input_df)[:, 1][0])
            explanation = get_shap_values(model, input_df)
            fig = get_waterfall_figure(model, input_df, feature_names, explanation=explanation)
            factors = get_top_factors(model, input_df, feature_names, n=3, explanation=explanation)
            tip = get_counterfactual(model, input_df, feature_names, explanation=explanation)

            st.session_state["prediction"] = {
                "probability": probability,
                "fig": fig,
                "factors": factors,
                "tip": tip,
            }

        with col_results:
            if "prediction" in st.session_state:
                pred = st.session_state["prediction"]

                render_probability_gauge(pred["probability"])

                st.markdown("#### SHAP Explanation")
                st.pyplot(pred["fig"])
                plt.close(pred["fig"])

                render_top_factors(pred["factors"])

                render_counterfactual(pred["tip"])
            else:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 60px 20px; color: #858585;">
                        <p style="font-size: 1.5rem;">_</p>
                        <p>Fill in your profile and click <b>Predict</b><br/>
                        to see your re-shopping score.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab2:
    render_quality_tab()
