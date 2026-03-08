"""
Page 4 — Prediction (Live Interface)
Accept user inputs, run the best trained model, display result and risk score.
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import CATEGORIES, build_single_row

st.title("Live Fraud Prediction")
st.markdown("---")

# ---- Check for trained models ----
if "trained_models" not in st.session_state or not st.session_state["trained_models"]:
    st.warning(
        "No models have been trained yet. "
        "Please go to the **Model Training** page and train at least one model first."
    )
    st.stop()

# ---- Pick best model (highest Recall) ----
metrics = st.session_state.get("model_metrics", {})
best_name = max(metrics, key=lambda m: metrics[m].get("Recall", 0))
best_model = st.session_state["trained_models"][best_name]

st.info(f"Using **{best_name}** (Recall = {metrics[best_name]['Recall']:.4f}) for predictions.")

# ---- Input form ----
st.markdown("### Enter Transaction Details")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=30000.0,
                              value=50.0, step=1.0)
        category = st.selectbox("Merchant Category", CATEGORIES)
    with col2:
        hour = st.slider("Transaction Hour (0–23)", 0, 23, 12)
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)

    # Advanced (optional) — collapsed
    with st.expander("Advanced Options (optional)"):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            day_of_week = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                index=2,
            )
            month = st.slider("Month", 1, 12, 6)
        with adv_col2:
            gender = st.selectbox("Gender", ["F", "M"])
            city_pop = st.number_input("City Population", min_value=100,
                                       max_value=5_000_000, value=50_000, step=1000)
            distance_km = st.number_input("Distance to Merchant (km)", min_value=0.0,
                                          max_value=500.0, value=30.0, step=1.0)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ---- Prediction output ----
if submitted:
    row = build_single_row(
        amt=amt, category=category, hour=hour, age=age,
        day_of_week=day_of_week, month=month,
        city_pop=city_pop, distance_km=distance_km, gender=gender,
    )

    proba = best_model.predict_proba(row)[0][1]
    prediction = int(proba >= 0.5)
    risk_pct = proba * 100

    st.markdown("---")
    st.markdown("### Prediction Result")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 0:
            st.success("**Legitimate Transaction**")
        else:
            st.error("**Suspicious — Potential Fraud!**")

    with res_col2:
        st.metric("Risk Score", f"{risk_pct:.1f}%")

    # Visual gauge
    st.markdown("#### Risk Gauge")
    st.progress(min(proba, 1.0))

    if risk_pct < 25:
        st.caption("Low risk — transaction appears normal.")
    elif risk_pct < 50:
        st.caption("Moderate risk — consider additional verification.")
    elif risk_pct < 75:
        st.caption("High risk — manual review recommended.")
    else:
        st.caption("Very high risk — strong indicators of fraud.")
