"""
Fraud Transaction Detector — Streamlit Application
Entry point. Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Fraud Transaction Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar branding ----
st.sidebar.image("https://img.icons8.com/fluency/96/security-checked.png", width=72)
st.sidebar.title("🛡️ Fraud Detector")
st.sidebar.markdown("---")
st.sidebar.caption("Navigate using the pages above.")

# ---- Landing content (shown when no page is selected) ----
st.title("🛡️ Fraud Transaction Detector")
st.markdown(
    """
    Welcome to the **Fraud Transaction Detector** dashboard.  
    Use the **sidebar** to navigate between pages:

    | Page | Description |
    |------|-------------|
    | **Home** | Project overview & dataset summary |
    | **EDA** | Interactive exploratory data analysis |
    | **Model Training** | Train & tune ML models |
    | **Prediction** | Real-time fraud prediction |
    | **Model Comparison** | Side-by-side model evaluation |
    """
)
