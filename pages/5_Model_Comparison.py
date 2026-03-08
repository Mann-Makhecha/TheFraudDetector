"""
Page 5 — Model Comparison
Side-by-side comparison of all trained models with bar charts and best-model callout.
"""

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.visualizations import plot_model_comparison

st.title("Model Comparison")
st.markdown("---")

# ---- Check for trained models ----
metrics = st.session_state.get("model_metrics", {})

if len(metrics) == 0:
    st.warning(
        "No models have been trained yet. "
        "Go to the **Model Training** page and train at least one model."
    )
    st.stop()

if len(metrics) == 1:
    st.info(
        "Only one model has been trained so far. "
        "Train additional models for a meaningful comparison."
    )

# ---- Metrics table ----
st.markdown("### Metrics Summary")
metrics_df = pd.DataFrame(metrics).T
metrics_df.index.name = "Model"
metrics_df = metrics_df[["Precision", "Recall", "F1-Score", "ROC-AUC"]]

# Highlight best per column
def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_max]

st.dataframe(
    metrics_df.style.apply(highlight_max, axis=0).format("{:.4f}"),
    use_container_width=True,
)

# ---- Grouped bar chart ----
st.markdown("### Visual Comparison")
fig = plot_model_comparison(metrics)
st.pyplot(fig)

# ---- Best model callout ----
st.markdown("---")
st.markdown("### Best Model Selection")

best_name = max(metrics, key=lambda m: metrics[m].get("Recall", 0))
best_recall = metrics[best_name]["Recall"]

st.success(f"**Best Model: {best_name}** — Recall: **{best_recall:.4f}**")
st.markdown(
    f"""
    **Why {best_name}?**

    The primary business objective is to **catch as many fraudulent transactions as possible**
    to minimise financial losses. **Recall** measures exactly this — the proportion of actual
    fraud cases that the model correctly identifies.

    A higher Recall means fewer fraudulent transactions slip through undetected. While this
    may come at the cost of slightly more false positives (lower Precision), the financial
    impact of missing fraud far outweighs the inconvenience of flagging a few legitimate
    transactions for review.

    **{best_name}** achieved the highest Recall of **{best_recall:.4f}** ({best_recall*100:.2f}%)
    among all trained models, making it the recommended choice for production deployment.
    """
)

# ---- Individual model details ----
st.markdown("---")
st.markdown("### Model Details")
params = st.session_state.get("model_params", {})
for mname in metrics:
    with st.expander(f"{mname}"):
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("**Hyperparameters**")
            st.json(params.get(mname, {}))
        with dcol2:
            st.markdown("**Metrics**")
            for k, v in metrics[mname].items():
                st.metric(k, f"{v:.4f}")
