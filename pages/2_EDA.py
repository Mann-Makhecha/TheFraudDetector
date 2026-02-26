"""
Page 2 — Exploratory Data Analysis (EDA)
Interactive dashboard with 7 visualisations, sidebar filters, and insights.
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data, engineer_features
from utils.visualizations import (
    plot_fraud_countplot,
    plot_fraud_rate_by_category,
    plot_amount_histogram,
    plot_amount_boxplot,
    plot_fraud_heatmap,
    plot_top_states,
    plot_fraud_over_time,
)

st.title("📊 Exploratory Data Analysis")
st.markdown("---")

# ---- Load & engineer ----
raw_df = load_data()
df = engineer_features(raw_df)

# ---- Sidebar filters ----
st.sidebar.header("🔍 EDA Filters")

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

# Apply filters
mask = (
    (df["category"].isin(selected_categories))
    & (df["amt"] >= amt_range[0])
    & (df["amt"] <= amt_range[1])
    & (df["state"].isin(selected_states))
)
filtered = df[mask]

st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** transactions after filters.")

# ---- Chart 1: Countplot ----
st.subheader("1️⃣ Fraud vs. Legitimate Distribution")
st.pyplot(plot_fraud_countplot(filtered))
fraud_pct = filtered["is_fraud"].mean() * 100
st.caption(
    f"📌 **Insight:** Fraud represents **{fraud_pct:.2f}%** of filtered transactions — "
    "a severe class imbalance that must be addressed during model training."
)

st.markdown("---")

# ---- Chart 2: Fraud rate by category ----
st.subheader("2️⃣ Fraud Rate by Merchant Category")
st.pyplot(plot_fraud_rate_by_category(filtered))
st.caption(
    "📌 **Insight:** Categories like **shopping_net** and **misc_net** tend to show "
    "the highest fraud rates, likely due to card-not-present (online) transactions."
)

st.markdown("---")

# ---- Chart 3: Amount histogram ----
st.subheader("3️⃣ Transaction Amount Distribution")
st.pyplot(plot_amount_histogram(filtered))
st.caption(
    "📌 **Insight:** Fraudulent transactions skew toward **higher amounts** compared "
    "to legitimate ones, though fraud also occurs at small amounts."
)

st.markdown("---")

# ---- Chart 4: Amount box plot ----
st.subheader("4️⃣ Transaction Amount by Fraud Label")
st.pyplot(plot_amount_boxplot(filtered))
st.caption(
    "📌 **Insight:** The median transaction amount for fraud is noticeably **higher** "
    "with a wider interquartile range, confirming amount as a useful feature."
)

st.markdown("---")

# ---- Chart 5: Heatmap ----
st.subheader("5️⃣ Fraud by Hour of Day vs. Day of Week")
st.pyplot(plot_fraud_heatmap(filtered))
st.caption(
    "📌 **Insight:** Late-night and early-morning hours (roughly **22:00–04:00**) "
    "show elevated fraud activity across most days of the week."
)

st.markdown("---")

# ---- Chart 6: Top states ----
st.subheader("6️⃣ Top States by Fraud Count")
top_n = st.slider("Number of states to show", 5, 30, 15)
st.pyplot(plot_top_states(filtered, top_n=top_n))
st.caption(
    "📌 **Insight:** Populous states such as **TX, NY, and CA** dominate fraud volume "
    "purely due to higher transaction counts."
)

st.markdown("---")

# ---- Chart 7: Fraud over time ----
st.subheader("7️⃣ Fraud Transaction Volume Over Time")
st.pyplot(plot_fraud_over_time(filtered))
st.caption(
    "📌 **Insight:** Fraud volume remains relatively **steady** over the observed period "
    "with slight seasonal fluctuations."
)
