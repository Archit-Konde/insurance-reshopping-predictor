"""Results panel component for displaying predictions and explanations."""

import streamlit as st


def render_probability_gauge(probability: float):
    """Render a large probability score with color-coded risk tier.

    Args:
        probability: Re-shopping probability (0-1).
    """
    pct = probability * 100

    if pct > 60:
        color = "#28c840"
        tier = "Likely to save"
        tier_icon = "+"
    elif pct > 30:
        color = "#C9A84C"
        tier = "Possibly save"
        tier_icon = "~"
    else:
        color = "#858585"
        tier = "Unlikely to save"
        tier_icon = "-"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; background: #252526;
                    border-radius: 10px; border: 1px solid #3e3e42; margin-bottom: 20px;">
            <div style="font-size: 3.5rem; font-weight: bold; color: {color};
                        font-family: 'JetBrains Mono', monospace;">
                {pct:.1f}%
            </div>
            <div style="font-size: 1.2rem; color: {color}; margin-top: 5px;
                        font-family: 'JetBrains Mono', monospace;">
                [{tier_icon}] {tier}
            </div>
            <div style="font-size: 0.85rem; color: #858585; margin-top: 10px;">
                Re-shopping probability score
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_factors(factors: list):
    """Render top contributing factors as styled cards.

    Args:
        factors: List of dicts from explain.get_top_factors().
    """
    st.markdown("#### Key Factors")

    for factor in factors:
        if factor["direction"] == "increases":
            arrow = "^"
            color = "#28c840"
        else:
            arrow = "v"
            color = "#C9A84C"

        st.markdown(
            f"""
            <div style="padding: 12px 16px; background: #252526;
                        border-left: 3px solid {color}; border-radius: 4px;
                        margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;">
                <span style="color: {color}; font-weight: bold;">[{arrow}]</span>
                <span style="color: #d4d4d4; font-weight: bold;"> {factor['label']}</span>
                <br/>
                <span style="color: #858585; font-size: 0.85rem;">
                    {factor['plain_english']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_counterfactual(tip: str):
    """Render counterfactual suggestion in a callout box."""
    st.markdown(
        f"""
        <div style="padding: 16px; background: #1a2332; border: 1px solid #2a4a6b;
                    border-radius: 8px; margin-top: 16px;
                    font-family: 'JetBrains Mono', monospace;">
            <div style="color: #569cd6; font-weight: bold; margin-bottom: 8px;">
                [i] What could change your score?
            </div>
            <div style="color: #d4d4d4; font-size: 0.9rem;">
                {tip}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
