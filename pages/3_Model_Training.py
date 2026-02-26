"""
Page 3 — Model Training
Train Logistic Regression, Decision Tree, or Random Forest with tuneable
hyperparameters. Results are stored in session_state for the Comparison page.
"""

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data, engineer_features, get_model_features
from utils.model_utils import train_model, evaluate_model, save_model
from utils.visualizations import plot_confusion_matrix

st.title("🤖 Model Training")
st.markdown("---")

# ---- Initialise session state ----
if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = {}   # {name: model}
if "model_metrics" not in st.session_state:
    st.session_state["model_metrics"] = {}    # {name: metrics_dict}
if "model_params" not in st.session_state:
    st.session_state["model_params"] = {}     # {name: params_dict}

# ---- Sidebar controls ----
st.sidebar.header("⚙️ Training Settings")

model_name = st.sidebar.radio(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Random Forest"],
)

test_size = st.sidebar.slider("Test Size", 0.10, 0.40, 0.20, 0.05)

params = {}
if model_name == "Logistic Regression":
    params["C"] = st.sidebar.slider("Regularisation C", 0.01, 10.0, 1.0, 0.01)
    params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 1000, 500, 50)
elif model_name == "Decision Tree":
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 30, 10)
    params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 20, 5)
elif model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Number of Estimators", 50, 500, 100, 10)
    params["max_depth"] = st.sidebar.slider("Max Depth", 5, 30, 15)
    params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 10, 5)

# ---- Data preparation ----
@st.cache_data(show_spinner="Preparing training data…")
def prepare_data(test_size_val):
    raw = load_data()
    df = engineer_features(raw)
    X, y, feature_names = get_model_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_val, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names

# ---- Train button ----
st.markdown(f"### Training: **{model_name}**")
st.markdown("**Hyperparameters:**")
st.json(params)

if st.button("🚀 Train Model", type="primary", use_container_width=True):
    progress = st.progress(0, text="Loading data…")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(test_size)
    progress.progress(20, text="Data ready. Training model…")

    with st.spinner(f"Training {model_name}…"):
        model, elapsed = train_model(model_name, X_train, y_train, params)
    progress.progress(70, text="Evaluating model…")

    metrics, cm, report = evaluate_model(model, X_test, y_test)
    progress.progress(100, text="Done ✅")

    # Persist results
    save_model(model, model_name)
    st.session_state["trained_models"][model_name] = model
    st.session_state["model_metrics"][model_name] = metrics
    st.session_state["model_params"][model_name] = params
    st.session_state["feature_names"] = feature_names

    # ---- Display results ----
    st.success(f"✅ **{model_name}** trained in **{elapsed:.2f}s**")

    st.markdown("#### 📈 Performance Metrics")
    met_cols = st.columns(4)
    for i, (k, v) in enumerate(metrics.items()):
        met_cols[i].metric(k, f"{v:.4f}")

    st.markdown("#### 🔢 Confusion Matrix")
    st.pyplot(plot_confusion_matrix(cm))

    st.markdown("#### 📋 Classification Report")
    st.code(report)

# ---- Show previously trained models ----
if st.session_state["model_metrics"]:
    st.markdown("---")
    st.markdown("### 📦 Trained Models in This Session")
    for mname, mmetrics in st.session_state["model_metrics"].items():
        with st.expander(f"{mname}"):
            st.json({"params": st.session_state["model_params"].get(mname, {}),
                      "metrics": mmetrics})
