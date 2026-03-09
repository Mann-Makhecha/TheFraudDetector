"""
Page 5 — Model Comparison
Side-by-side comparison of all trained models with bar charts, ROC curves,
PR curves, and best-model callout. Models are pre-loaded from session state.
"""

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.visualizations import (
    plot_model_comparison,
    plot_roc_curves,
    plot_precision_recall_curves,
)

st.title(" Model Comparison")
st.markdown("---")

# ---- Check for results ----
all_results = st.session_state.get("all_results")
if not all_results:
    st.warning(
        "Models have not been loaded yet. "
        "Please go to the **main page** first to trigger model loading."
    )
    st.stop()

metrics = {name: res["metrics"] for name, res in all_results.items()}

# ---- Sidebar controls ----
st.sidebar.header("Comparison Settings")

comparison_metric = st.sidebar.selectbox(
    "Sort / Highlight By",
    ["Recall", "Precision", "F1-Score", "ROC-AUC"],
    index=0,
    help="Choose the primary metric for highlighting the best model.",
)

show_roc_comp = st.sidebar.checkbox("Show ROC Curves", value=True)
show_pr_comp = st.sidebar.checkbox("Show Precision-Recall Curves", value=True)
show_training_time = st.sidebar.checkbox("Show Training Times", value=True)

# =====================================================================
# Metrics Table
# =====================================================================
st.markdown("###  Metrics Summary")
metrics_df = pd.DataFrame(metrics).T
metrics_df.index.name = "Model"
metrics_df = metrics_df[["Precision", "Recall", "F1-Score", "ROC-AUC"]]

# Add training time column
metrics_df["Training Time (s)"] = [
    f"{all_results[name]['elapsed']:.2f}" for name in metrics_df.index
]

# Highlight best per column
def highlight_max(s):
    try:
        numeric = pd.to_numeric(s, errors="coerce")
        is_max = numeric == numeric.max()
        return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_max]
    except Exception:
        return ["" for _ in s]

st.dataframe(
    metrics_df.style.apply(highlight_max, axis=0).format(
        {
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
        }
    ),
    use_container_width=True,
)

# =====================================================================
# Grouped Bar Chart
# =====================================================================
st.markdown("###  Visual Comparison")
fig = plot_model_comparison(metrics)
st.pyplot(fig)

st.markdown("---")

# =====================================================================
# ROC Curves
# =====================================================================
if show_roc_comp:
    st.markdown("###  ROC Curves Overlay")
    st.pyplot(plot_roc_curves(all_results))
    st.markdown("---")

# =====================================================================
# Precision-Recall Curves
# =====================================================================
if show_pr_comp:
    st.markdown("###  Precision-Recall Curves Overlay")
    st.pyplot(plot_precision_recall_curves(all_results))
    st.markdown("---")

# =====================================================================
# Training Times
# =====================================================================
if show_training_time:
    st.markdown("### ⏱️ Training Times")
    time_cols = st.columns(len(all_results))
    for i, (name, res) in enumerate(all_results.items()):
        with time_cols[i]:
            st.metric(name, f"{res['elapsed']:.2f}s")

    st.markdown("---")

# =====================================================================
# Best Model Callout
# =====================================================================
st.markdown("###  Best Model Selection")

best_name = max(metrics, key=lambda m: metrics[m].get(comparison_metric, 0))
best_val = metrics[best_name][comparison_metric]

st.success(f"**Best Model: {best_name}** — {comparison_metric}: **{best_val:.4f}**")
st.markdown(
    f"""
    **Why {best_name}?**

    Based on the selected metric **{comparison_metric}**, **{best_name}** achieved the highest
    score of **{best_val:.4f}** ({best_val*100:.2f}%) among all trained models.

    {"**Recall** measures the proportion of actual fraud cases correctly identified. A higher Recall means fewer fraudulent transactions slip through undetected." if comparison_metric == "Recall" else ""}
    {"**Precision** measures how many flagged transactions are actually fraud. Higher Precision means fewer false alarms." if comparison_metric == "Precision" else ""}
    {"**F1-Score** is the harmonic mean of Precision and Recall, balancing both objectives." if comparison_metric == "F1-Score" else ""}
    {"**ROC-AUC** measures overall discriminative ability across all thresholds." if comparison_metric == "ROC-AUC" else ""}
    """
)

# =====================================================================
# Individual Model Details
# =====================================================================
st.markdown("---")
st.markdown("###  Model Details")
for mname in metrics:
    with st.expander(f" {mname}"):
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("**Hyperparameters**")
            st.json(all_results[mname]["params"])
        with dcol2:
            st.markdown("**Metrics**")
            for k, v in metrics[mname].items():
                st.metric(k, f"{v:.4f}")
        st.markdown("**Classification Report**")
        st.code(all_results[mname]["report"])
