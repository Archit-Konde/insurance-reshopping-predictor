"""
Explainability module for Insurance Re-Shopping Predictor.

Uses SHAP TreeExplainer for LightGBM — fast exact Shapley values
for tree-based models. Provides:
- SHAP waterfall charts for individual predictions
- Top contributing factors in plain English
- Counterfactual suggestions (single most actionable change)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# Human-readable feature names for the UI
FEATURE_LABELS = {
    "Gender": "Gender",
    "Age": "Age",
    "Driving_License": "Driving license status",
    "Region_Code": "Region",
    "Previously_Insured": "Previously insured",
    "Vehicle_Age": "Vehicle age",
    "Vehicle_Damage": "Vehicle damage history",
    "Annual_Premium_log": "Annual premium",
    "Policy_Sales_Channel": "Policy sales channel",
    "Vintage": "Months as customer",
}

# Features the user can realistically change (for counterfactual)
ACTIONABLE_FEATURES = {
    "Previously_Insured": "switching your insurance status",
    "Vehicle_Damage": "your vehicle damage history",
    "Annual_Premium_log": "adjusting your premium",
    "Policy_Sales_Channel": "using a different sales channel",
    "Vehicle_Age": "as your vehicle ages",
}


def get_shap_explainer(model):
    """Create a SHAP TreeExplainer for the LightGBM model."""
    return shap.TreeExplainer(model)


def get_shap_values(model, input_df):
    """Compute SHAP values for a single input.

    Args:
        model: Trained LightGBM model.
        input_df: Single-row DataFrame of preprocessed features.

    Returns:
        shap.Explanation object with values, base_values, and data.
    """
    explainer = get_shap_explainer(model)
    shap_values = explainer(input_df)

    # For binary classification, use class 1 (re-shop) SHAP values
    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]
    return shap_values


def get_waterfall_figure(model, input_df, feature_names, explanation=None):
    """Generate a SHAP waterfall chart for a single prediction.

    Args:
        model: Trained LightGBM model.
        input_df: Single-row preprocessed DataFrame.
        feature_names: List of feature column names.
        explanation: Pre-computed SHAP explanation (avoids recomputation).

    Returns:
        matplotlib Figure object.
    """
    if explanation is None:
        explanation = get_shap_values(model, input_df)

    # Use human-readable labels
    display_names = [FEATURE_LABELS.get(f, f) for f in feature_names]
    explanation.feature_names = display_names

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(explanation[0], show=False, max_display=10)
    plt.title("What drives your re-shopping score", fontsize=14, pad=15)
    plt.tight_layout()

    return fig


def get_top_factors(model, input_df, feature_names, n=3, explanation=None):
    """Get the top N contributing factors in plain English.

    Args:
        model: Trained LightGBM model.
        input_df: Single-row preprocessed DataFrame.
        feature_names: List of feature column names.
        n: Number of top factors to return.
        explanation: Pre-computed SHAP explanation (avoids recomputation).

    Returns:
        List of dicts with keys: feature, direction, magnitude, plain_english.
    """
    if explanation is None:
        explanation = get_shap_values(model, input_df)
    shap_vals = explanation[0].values

    # Sort by absolute magnitude
    indices = np.argsort(np.abs(shap_vals))[::-1][:n]

    factors = []
    for idx in indices:
        feature = feature_names[idx]
        value = float(shap_vals[idx])
        label = FEATURE_LABELS.get(feature, feature)
        direction = "increases" if value > 0 else "decreases"
        magnitude = abs(value)

        # Convert SHAP value (log-odds) to approximate percentage impact
        pct = round(magnitude * 100, 0)

        plain_english = (
            f"Your {label.lower()} {direction} your "
            f"re-shopping likelihood by {pct:.0f}%"
        )

        factors.append({
            "feature": feature,
            "label": label,
            "direction": direction,
            "magnitude": round(magnitude, 4),
            "pct": pct,
            "plain_english": plain_english,
        })

    return factors


def get_counterfactual(model, input_df, feature_names, explanation=None):
    """Find the single most actionable change to move the score.

    Looks only at actionable features (things the user could realistically
    change or that naturally change over time).

    Args:
        model: Trained LightGBM model.
        input_df: Single-row preprocessed DataFrame.
        feature_names: List of feature column names.
        explanation: Pre-computed SHAP explanation (avoids recomputation).

    Returns:
        String with the counterfactual suggestion.
    """
    if explanation is None:
        explanation = get_shap_values(model, input_df)
    shap_vals = explanation[0].values

    # Find the actionable feature with the largest negative SHAP value
    # (i.e., the factor most reducing the score that could be changed)
    best_feature = None
    best_impact = 0

    for idx, feature in enumerate(feature_names):
        if feature in ACTIONABLE_FEATURES:
            impact = abs(float(shap_vals[idx]))
            if impact > best_impact:
                best_impact = impact
                best_feature = feature

    if best_feature is None:
        return "Your profile factors are mostly non-actionable (age, region). Consider checking back as your vehicle ages."

    feature_idx = feature_names.index(best_feature)
    shap_val = float(shap_vals[feature_idx])
    action = ACTIONABLE_FEATURES[best_feature]
    pct = round(abs(shap_val) * 100, 0)

    if shap_val < 0:
        return (
            f"The biggest opportunity: {action} currently reduces your score by "
            f"~{pct:.0f}%. Changing this factor could significantly increase your "
            f"re-shopping benefit."
        )
    else:
        return (
            f"{action.capitalize()} is already working in your favor (+{pct:.0f}%). "
            f"Your current profile suggests you're well-positioned to benefit from re-shopping."
        )
