"""
Fraud Transaction Detector — Streamlit Application
Entry point. Run with:  streamlit run app.py

Train-once architecture: all models are trained on first launch,
cached to disk, and loaded instantly on subsequent runs.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import get_engineered_train, get_engineered_test, get_model_features
from utils.model_utils import load_all_results, train_all_models

st.set_page_config(
    page_title="Fraud Transaction Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Auto-train models once (on first launch) ----
if "all_results" not in st.session_state:
    cached = load_all_results()
    if cached is not None:
        st.session_state["all_results"] = cached
        # Build metrics dict for backward compat
        st.session_state["model_metrics"] = {
            name: res["metrics"] for name, res in cached.items()
        }
        st.session_state["model_params"] = {
            name: res["params"] for name, res in cached.items()
        }
        st.session_state["trained_models"] = {
            name: res["model"] for name, res in cached.items()
        }
    else:
        # Need to train — show progress
        st.title("Fraud Transaction Detector")
        st.markdown("### First-time setup: Training all models...")
        st.info("This only happens **once**. Models will be cached to disk for future launches.")

        progress = st.progress(0, text="Loading training data…")

        # Load and prepare data
        progress.progress(10, text="Loading and preparing training data…")
        df_train = get_engineered_train()
        X_train_full, y_train_full, feature_names = get_model_features(df_train)

        progress.progress(30, text="Loading and preparing test data…")
        df_test = get_engineered_test()
        X_test, y_test, _ = get_model_features(df_test)

        progress.progress(35, text="Starting model training…")

        def _progress_cb(pct, text):
            # Scale 0-100 to 35-95
            scaled = 35 + int(pct * 0.60)
            progress.progress(min(scaled, 95), text=text)

        all_results = train_all_models(
            X_train_full, X_test, y_train_full, y_test,
            feature_names, progress_callback=_progress_cb,
        )
        progress.progress(100, text="All models trained and cached!")

        st.session_state["all_results"] = all_results
        st.session_state["model_metrics"] = {
            name: res["metrics"] for name, res in all_results.items()
        }
        st.session_state["model_params"] = {
            name: res["params"] for name, res in all_results.items()
        }
        st.session_state["trained_models"] = {
            name: res["model"] for name, res in all_results.items()
        }
        st.session_state["feature_names"] = feature_names

        st.success("**All 3 models trained and saved!** Navigate to any page using the sidebar.")


# ---- App Navigation Customization ----
# Using st.navigation allows hiding the main entry point ("app.py") from the sidebar

pages = {
    "Dashboard": [
        st.Page("pages/1_Home.py", title="Home", default=True),
        st.Page("pages/2_EDA.py", title="EDA"),
        st.Page("pages/3_Model_Results.py", title="Model Results"),
        st.Page("pages/4_Prediction.py", title="Prediction"),
        st.Page("pages/5_Model_Comparison.py", title="Model Comparison"),
    ]
}

pg = st.navigation(pages)

# Sidebar styling
st.sidebar.markdown("## Fraud Detector")
st.sidebar.caption("Detection & Analysis Platform")

# Run the selected page
pg.run()
