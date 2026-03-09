"""
Page 1 — Home
Comprehensive project overview: dataset details, app purpose, use cases,
pros & cons, how it works, feature engineering, team, and tech stack.
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data
from utils.visualizations import plot_class_distribution_pie

st.title(" Home")
st.markdown("---")

# Hero Section
st.markdown(
    """
    ##  Fraud Transaction Detector
    > *Identifying fraudulent credit card transactions using Machine Learning*

    Payment fraud causes **massive financial losses** globally — estimated at over
    **$30 billion annually**. This application leverages supervised machine learning
    to flag suspicious transactions in real time, helping financial institutions
    minimise losses and protect their customers.
    """
)

# About This Application
st.markdown("---")
st.markdown("###  About This Application")
st.markdown(
    """
    The **Fraud Transaction Detector** is an end-to-end machine learning dashboard built
    with **Streamlit**. It analyses a large simulated dataset of credit card transactions
    to detect fraudulent activity using three supervised learning models.

    **Key Highlights:**
    -  **Train-Once Architecture** — Models are automatically trained on the first launch
      and cached to disk. No manual re-training needed.
    -  **Interactive EDA** — 7 visualisations with sidebar filters for deep data exploration.
    -  **3 ML Models** — Logistic Regression, Decision Tree, and Random Forest.
    -  **Optimised for Recall** — The priority is catching fraud, not minimising false alarms.
    -  **Live Predictions** — Enter transaction details and get an instant risk score.
    -  **Model Comparison** — Side-by-side evaluation with ROC and PR curves.
    """
)

# Dataset Overview
st.markdown("---")
st.markdown("###  Dataset Overview")

df = load_data()

total_rows = len(df)
total_features = df.shape[1]
fraud_count = int(df["is_fraud"].sum())
legit_count = total_rows - fraud_count
fraud_pct = fraud_count / total_rows * 100
date_min = df["trans_date_trans_time"].min().strftime("%Y-%m-%d")
date_max = df["trans_date_trans_time"].max().strftime("%Y-%m-%d")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_rows:,}")
col2.metric("Features", total_features)
col3.metric("Fraud Transactions", f"{fraud_count:,}  ({fraud_pct:.2f}%)")
col4.metric("Date Range", f"{date_min}  →  {date_max}")

# Two-column layout: table + pie chart
ds_col1, ds_col2 = st.columns([1.2, 1])

with ds_col1:
    st.markdown(
        """
        | Property | Value |
        |----------|-------|
        | **Source** | [Kaggle — Credit Card Fraud (Simulated)](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
        | **Training File** | `fraudTrain.csv` (~1.3M rows) |
        | **Test File** | `fraudTest.csv` (~555K rows) |
        | **Target Variable** | `is_fraud` (0 = Legitimate, 1 = Fraud) |
        | **Key Challenge** | Severe class imbalance (~0.58% fraud) |
        | **Time Period** | Jan 2019 – Dec 2020 |
        | **Geography** | United States |
        """
    )

with ds_col2:
    st.pyplot(plot_class_distribution_pie(fraud_count, legit_count))

# Column descriptions
with st.expander(" Column Descriptions (click to expand)"):
    st.markdown(
        """
        | Column | Type | Description |
        |--------|------|-------------|
        | `trans_date_trans_time` | datetime | Timestamp of the transaction |
        | `category` | string | Merchant category (14 types) |
        | `amt` | float | Transaction amount in USD |
        | `gender` | string | Cardholder gender (M/F) |
        | `city` | string | Cardholder city |
        | `state` | string | Cardholder state |
        | `zip` | int | ZIP code |
        | `lat` / `long` | float | Cardholder lat/long |
        | `city_pop` | int | City population |
        | `job` | string | Cardholder occupation |
        | `dob` | date | Cardholder date of birth |
        | `merch_lat` / `merch_long` | float | Merchant lat/long |
        | `is_fraud` | int | Target: 0 = Legitimate, 1 = Fraud |
        """
    )

# Sample data
with st.expander(" Sample Rows (click to expand)"):
    st.dataframe(df.head(10), use_container_width=True)

# How It Works
st.markdown("---")
st.markdown("###  How It Works")

hw1, hw2, hw3, hw4 = st.columns(4)

with hw1:
    st.markdown("####  Load Data")
    st.markdown(
        "Load **1.3M+** transactions from CSV with optimised dtypes for fast processing."
    )

with hw2:
    st.markdown("####  Engineer Features")
    st.markdown(
        "Extract **hour**, **day**, **age**, **distance**, and **log-amount** from raw columns."
    )

with hw3:
    st.markdown("####  Train Models")
    st.markdown(
        "Train **3 models** with `class_weight='balanced'` to handle the extreme class imbalance."
    )

with hw4:
    st.markdown("####  Predict & Evaluate")
    st.markdown(
        "Evaluate on test data, display **ROC curves**, and provide **live fraud prediction**."
    )

# Use Cases
st.markdown("---")
st.markdown("###  Use Cases")

uc1, uc2 = st.columns(2)

with uc1:
    st.markdown(
        """
        ** Banking & Financial Institutions**
        - Real-time fraud detection on incoming transactions
        - Automated flagging for manual review queues
        - Reducing chargebacks and financial losses

        ** E-Commerce Platforms**
        - Screening high-risk online purchases
        - Card-not-present fraud detection
        - Protecting merchants from fraudulent orders
        """
    )

with uc2:
    st.markdown(
        """
        ** Insurance Companies**
        - Claims fraud detection
        - Pattern analysis for suspicious activity
        - Risk scoring for underwriting decisions

        ** Fintech & Payment Processors**
        - Scoring transactions in payment gateways
        - Integration with existing fraud rules engines
        - Building ML-augmented fraud pipelines
        """
    )

# Pros & Cons
st.markdown("---")
st.markdown("###  Pros &  Cons")

pc1, pc2 = st.columns(2)

with pc1:
    st.success("**Advantages**")
    st.markdown(
        """
        -  **High Recall** — Catches the majority of fraudulent transactions
        -  **Multiple Models** — Compare 3 algorithms to find the best fit
        -  **Balanced Training** — Handles 99.4% vs 0.6% class imbalance
        -  **Feature Engineering** — Distance, time, and age features boost accuracy
        -  **Interactive Dashboard** — Real-time filtering and predictions
        -  **Train-Once** — Models persist to disk; no re-training on restart
        -  **Large Dataset** — 1.3M training + 555K test records
        """
    )

with pc2:
    st.error("**Limitations**")
    st.markdown(
        """
        -  **Simulated Data** — Not real-world transaction data
        -  **No Deep Learning** — Limited to classical ML models
        -  **Static Features** — No real-time behavioural features
        -  **Binary Classification** — Does not categorise fraud types
        -  **No Concept Drift** — Model does not adapt over time
        -  **False Positives** — High recall may flag some legitimate transactions
        -  **US-Only** — Dataset is limited to US geography
        """
    )

# Feature Engineering
st.markdown("---")
st.markdown("###  Feature Engineering")
st.markdown("The following derived features are created from the raw data before training:")

st.markdown(
    """
    | Feature | Source | Description |
    |---------|--------|-------------|
    | `hour` | `trans_date_trans_time` | Hour of transaction (0–23) |
    | `day_of_week` | `trans_date_trans_time` | Day of week (0=Mon – 6=Sun) |
    | `month` | `trans_date_trans_time` | Month (1–12) |
    | `age` | `dob` + `trans_date_trans_time` | Customer age at transaction time |
    | `distance_km` | `lat/long` + `merch_lat/long` | Haversine distance between cardholder and merchant |
    | `amt_log` | `amt` | Log-transformed transaction amount |
    | `category_*` | `category` | One-hot encoded merchant category (14 columns) |
    | `gender_*` | `gender` | One-hot encoded gender (2 columns) |
    """
)

# Model Overview
st.markdown("---")
st.markdown("###  Models Used")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("#### Logistic Regression")
    st.markdown(
        """
        - **Type:** Linear classifier
        - **Strengths:** Fast, interpretable
        - **Key Params:** C, max_iter
        - **Best for:** Baseline performance
        """
    )

with m2:
    st.markdown("#### Decision Tree")
    st.markdown(
        """
        - **Type:** Tree-based classifier
        - **Strengths:** Non-linear, feature importance
        - **Key Params:** max_depth, min_samples_split
        - **Best for:** Interpretable rules
        """
    )

with m3:
    st.markdown("#### Random Forest")
    st.markdown(
        """
        - **Type:** Ensemble of trees
        - **Strengths:** Robust, high accuracy
        - **Key Params:** n_estimators, max_depth
        - **Best for:** Best overall performance
        """
    )

# Business Problem
st.markdown("---")
st.markdown("###  Business Problem")
st.info(
    "Credit card fraud is a growing threat in the digital economy. "
    "Traditional rule-based systems fail to keep pace with evolving fraud tactics. "
    "Machine learning models — optimised for **Recall** — can dramatically improve "
    "detection rates, catching fraudulent transactions that would otherwise slip through."
)

# Team Information
st.markdown("---")
st.markdown("###  Team Information")
team_data = {
    "Name": ["Samkit Jain", "Mann Makhecha", "Krish Jain", "Chirag Gupta", "Het Malvaniya", "Dhiraj Jagwani"],
}
st.table(team_data)

# Tech Stack
st.markdown("###  Tech Stack")
t1, t2, t3, t4, t5 = st.columns(5)
t1.markdown("![Python](https://img.icons8.com/color/28/python.png) **Python**")
t2.markdown("![Streamlit](https://img.icons8.com/color/28/streamlit.png) **Streamlit**")
t3.markdown("![Scikit](https://img.icons8.com/color/28/artificial-intelligence.png) **Scikit-learn**")
t4.markdown("![Pandas](https://img.icons8.com/color/28/pandas.png) **Pandas / NumPy**")
t5.markdown("![Chart](https://img.icons8.com/color/28/combo-chart.png) **Matplotlib / Seaborn**")
