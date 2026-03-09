"""
Page 2 — Exploratory Data Analysis (EDA)
Interactive dashboard with 7 visualisations, sidebar filters, and insights.
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import get_engineered_train
from utils.visualizations import (
    plot_fraud_countplot,
    plot_fraud_rate_by_category,
    plot_amount_histogram,
    plot_amount_boxplot,
    plot_fraud_heatmap,
    plot_top_states,
    plot_fraud_over_time,
)

st.title(" Exploratory Data Analysis")
st.markdown("---")

# ---- Load & engineer ----
df = get_engineered_train()

# ---- Sidebar filters ----
st.sidebar.header("EDA Filters")

all_categories = sorted(df["category"].unique())
selected_categories = st.sidebar.multiselect(
    "Merchant Category", all_categories, default=all_categories
)

amt_min, amt_max = float(df["amt"].min()), float(df["amt"].max())
amt_range = st.sidebar.slider(
    "Transaction Amount ($)",
    min_value=amt_min,
    max_value=amt_max,
    value=(amt_min, amt_max),
    step=1.0,
)

all_states = sorted(df["state"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "State", all_states, default=all_states
)

# Filter by fraud label
fraud_filter = st.sidebar.radio(
    "Show Transactions",
    ["All", "Fraud Only", "Legitimate Only"],
    index=0,
)

# Apply filters
mask = (
    (df["category"].isin(selected_categories))
    & (df["amt"] >= amt_range[0])
    & (df["amt"] <= amt_range[1])
    & (df["state"].isin(selected_states))
)
if fraud_filter == "Fraud Only":
    mask = mask & (df["is_fraud"] == 1)
elif fraud_filter == "Legitimate Only":
    mask = mask & (df["is_fraud"] == 0)

filtered = df[mask]

st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** transactions after filters.")

# ---- Chart selector ----
st.sidebar.header("Chart Settings")
chart_selection = st.sidebar.multiselect(
    "Select Charts to Display",
    [
        "1. Fraud vs. Legitimate Distribution",
        "2. Fraud Rate by Category",
        "3. Transaction Amount Distribution",
        "4. Transaction Amount by Label",
        "5. Fraud Heatmap (Hour × Day)",
        "6. Top States by Fraud Count",
        "7. Fraud Over Time",
    ],
    default=[
        "1. Fraud vs. Legitimate Distribution",
        "2. Fraud Rate by Category",
        "3. Transaction Amount Distribution",
        "4. Transaction Amount by Label",
        "5. Fraud Heatmap (Hour × Day)",
        "6. Top States by Fraud Count",
        "7. Fraud Over Time",
    ],
)

# ---- Chart 1: Countplot ----
if "1. Fraud vs. Legitimate Distribution" in chart_selection:
    st.subheader("1. Fraud vs. Legitimate Distribution")
    st.pyplot(plot_fraud_countplot(filtered))
    fraud_pct = filtered["is_fraud"].mean() * 100
    st.caption(
        f"**Insight:** Fraud represents **{fraud_pct:.2f}%** of filtered transactions — "
        "a severe class imbalance that must be addressed during model training."
    )
    st.markdown("---")

# ---- Chart 2: Fraud rate by category ----
if "2. Fraud Rate by Category" in chart_selection:
    st.subheader("2. Fraud Rate by Merchant Category")
    st.pyplot(plot_fraud_rate_by_category(filtered))
    st.caption(
        "**Insight:** Categories like **shopping_net** and **misc_net** tend to show "
        "the highest fraud rates, likely due to card-not-present (online) transactions."
    )
    st.markdown("---")

# ---- Chart 3: Amount histogram ----
if "3. Transaction Amount Distribution" in chart_selection:
    st.subheader("3. Transaction Amount Distribution")
    st.pyplot(plot_amount_histogram(filtered))
    st.caption(
        "**Insight:** Fraudulent transactions skew toward **higher amounts** compared "
        "to legitimate ones, though fraud also occurs at small amounts."
    )
    st.markdown("---")

# ---- Chart 4: Amount box plot ----
if "4. Transaction Amount by Label" in chart_selection:
    st.subheader("4. Transaction Amount by Fraud Label")
    st.pyplot(plot_amount_boxplot(filtered))
    st.caption(
        "**Insight:** The median transaction amount for fraud is noticeably **higher** "
        "with a wider interquartile range, confirming amount as a useful feature."
    )
    st.markdown("---")

# ---- Chart 5: Heatmap ----
if "5. Fraud Heatmap (Hour × Day)" in chart_selection:
    st.subheader("5. Fraud by Hour of Day vs. Day of Week")
    st.pyplot(plot_fraud_heatmap(filtered))
    st.caption(
        "**Insight:** Late-night and early-morning hours (roughly **22:00–04:00**) "
        "show elevated fraud activity across most days of the week."
    )
    st.markdown("---")

# ---- Chart 6: Top states ----
if "6. Top States by Fraud Count" in chart_selection:
    st.subheader("6. Top States by Fraud Count")
    top_n = st.slider("Number of states to show", 5, 30, 15)
    st.pyplot(plot_top_states(filtered, top_n=top_n))
    st.caption(
        "**Insight:** Populous states such as **TX, NY, and CA** dominate fraud volume "
        "purely due to higher transaction counts."
    )
    st.markdown("---")

# ---- Chart 7: Fraud over time ----
if "7. Fraud Over Time" in chart_selection:
    st.subheader("7. Fraud Transaction Volume Over Time")
    st.pyplot(plot_fraud_over_time(filtered))
    st.caption(
        "**Insight:** Fraud volume remains relatively **steady** over the observed period "
        "with slight seasonal fluctuations."
    )
