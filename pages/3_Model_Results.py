"""
Page 3 — Model Results
Displays pre-trained model metrics, confusion matrices, classification reports,
ROC curves, and feature importance charts. No re-training — models are loaded
from disk (trained on first launch via app.py).
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.visualizations import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_feature_importance,
)

st.title(" Model Results")
st.markdown("---")

# ---- Check for results ----
all_results = st.session_state.get("all_results")
if not all_results:
    st.warning(
        "Models have not been loaded yet. "
        "Please go to the **main page** (app.py) first to trigger model loading."
    )
    st.stop()

st.success(" All 3 models are pre-trained and loaded from disk. No re-training needed!")

# ---- Sidebar controls ----
st.sidebar.header("Model Results Settings")

model_names = list(all_results.keys())
selected_model = st.sidebar.radio("View Detailed Results For", model_names)

show_roc = st.sidebar.checkbox("Show ROC Curves", value=True)
show_pr = st.sidebar.checkbox("Show Precision-Recall Curves", value=True)
show_feat_imp = st.sidebar.checkbox("Show Feature Importance", value=True)
top_n_features = st.sidebar.slider("Top N Features", 5, 25, 15)

# =====================================================================
# Overview — All Models Summary
# =====================================================================
st.markdown("###  All Models — Performance Summary")

cols = st.columns(len(model_names))
for i, name in enumerate(model_names):
    res = all_results[name]
    with cols[i]:
        st.markdown(f"**{name}**")
        for metric, val in res["metrics"].items():
            st.metric(metric, f"{val:.4f}")
        st.caption(f"⏱️ Trained in {res['elapsed']:.2f}s")

st.markdown("---")

# =====================================================================
# ROC Curves
# =====================================================================
if show_roc:
    st.markdown("###  ROC Curves — All Models")
    st.pyplot(plot_roc_curves(all_results))
    st.caption(
        "**Insight:** The ROC curve shows the trade-off between true positive rate "
        "and false positive rate. A curve closer to the top-left corner indicates better performance."
    )
    st.markdown("---")

# =====================================================================
# Precision-Recall Curves
# =====================================================================
if show_pr:
    st.markdown("###  Precision-Recall Curves — All Models")
    st.pyplot(plot_precision_recall_curves(all_results))
    st.caption(
        "**Insight:** The PR curve is especially useful for imbalanced datasets. "
        "Higher area under the curve means better performance at distinguishing fraud."
    )
    st.markdown("---")

# =====================================================================
# Detailed Results for Selected Model
# =====================================================================
st.markdown(f"###  Detailed Results: **{selected_model}**")

res = all_results[selected_model]

# Metrics cards
st.markdown("#### Performance Metrics")
met_cols = st.columns(4)
for i, (k, v) in enumerate(res["metrics"].items()):
    met_cols[i].metric(k, f"{v:.4f}")

# Two-column: Confusion Matrix + Classification Report
detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    st.markdown("#### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(res["cm"]))

with detail_col2:
    st.markdown("#### Classification Report")
    st.code(res["report"])

# Hyperparameters
st.markdown("#### Hyperparameters Used")
st.json(res["params"])

st.markdown("---")

# =====================================================================
# Feature Importance
# =====================================================================
if show_feat_imp:
    st.markdown("###  Feature Importance")

    feat_imp = res.get("feature_importance")
    if feat_imp is not None:
        st.markdown(f"Showing top **{top_n_features}** features for **{selected_model}**:")
        fig = plot_feature_importance(feat_imp, top_n=top_n_features)
        if fig:
            st.pyplot(fig)
        st.caption(
            "**Insight:** Feature importance shows which variables the model relies on most. "
            "Higher values indicate greater influence on fraud prediction."
        )
    else:
        st.info(
            f"**{selected_model}** does not provide feature importance scores. "
            "Select a **Decision Tree** or **Random Forest** model to view this chart."
        )
