"""Input form component for the Insurance Re-Shopping Predictor."""

import streamlit as st


def render_input_form():
    """Render the prediction input form and return values dict on submit.

    Returns:
        dict with raw feature values if submitted, None otherwise.
    """
    st.markdown("### Your Insurance Profile")

    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    age = st.slider("Age", min_value=20, max_value=70, value=35)

    driving_license = st.checkbox("Driving License", value=True)

    region_code = st.selectbox(
        "Region Code",
        options=list(range(0, 53)),
        index=28,
        help="Geographic region (0-52)",
    )

    previously_insured = st.radio(
        "Previously Insured?",
        ["Yes", "No"],
        horizontal=True,
    )

    vehicle_age = st.selectbox(
        "Vehicle Age",
        ["< 1 Year", "1-2 Year", "> 2 Years"],
        index=1,
    )

    vehicle_damage = st.radio(
        "Vehicle Damage History",
        ["Yes", "No"],
        horizontal=True,
        help="Has the vehicle been damaged in the past?",
    )

    annual_premium = st.slider(
        "Current Annual Premium (INR)",
        min_value=20000,
        max_value=500000,
        value=80000,
        step=1000,
        format="%d",
    )

    policy_channel = st.slider(
        "Policy Sales Channel",
        min_value=1,
        max_value=163,
        value=26,
        help="Channel through which the policy was sold",
    )

    vintage = st.slider(
        "Vintage (Months as Customer)",
        min_value=10,
        max_value=300,
        value=150,
    )

    submitted = st.button("Predict", type="primary", use_container_width=True)

    if submitted:
        return {
            "Gender": gender,
            "Age": age,
            "Driving_License": 1 if driving_license else 0,
            "Region_Code": float(region_code),
            "Previously_Insured": 1 if previously_insured == "Yes" else 0,
            "Vehicle_Age": vehicle_age,
            "Vehicle_Damage": vehicle_damage,
            "Annual_Premium": float(annual_premium),
            "Policy_Sales_Channel": float(policy_channel),
            "Vintage": vintage,
        }

    return None
