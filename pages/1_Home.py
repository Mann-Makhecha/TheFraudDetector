"""
Page 1 — Home
Professional project overview, dataset description, team info, and business problem.
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data

st.title("Home")
st.markdown("---")

# ---- Hero Section ----
st.markdown(
    """
    ## Fraud Transaction Detector
    > *Identifying fraudulent credit card transactions using Machine Learning*

    Payment fraud causes **massive financial losses** globally — estimated at over
    **$30 billion annually**. This application leverages supervised machine learning
    to flag suspicious transactions in real time, helping financial institutions
    minimise losses and protect their customers.
    """
)

# ---- Dataset Summary ----
st.markdown("### Dataset Overview")

df = load_data()

total_rows = len(df)
total_features = df.shape[1]
fraud_count = int(df["is_fraud"].sum())
fraud_pct = fraud_count / total_rows * 100
date_min = df["trans_date_trans_time"].min().strftime("%Y-%m-%d")
date_max = df["trans_date_trans_time"].max().strftime("%Y-%m-%d")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_rows:,}")
col2.metric("Features", total_features)
col3.metric("Fraud Transactions", f"{fraud_count:,}  ({fraud_pct:.2f}%)")
col4.metric("Date Range", f"{date_min}  →  {date_max}")

st.markdown(
    """
    | Property | Value |
    |----------|-------|
    | **Source** | [Kaggle — Credit Card Fraud (Simulated)](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
    | **File Used** | `fraudTrain.csv` |
    | **Target Variable** | `is_fraud` (0 = Legitimate, 1 = Fraud) |
    | **Key Challenge** | Severe class imbalance (~0.58 % fraud) |
    """
)

# ---- Business Problem ----
st.markdown("### Business Problem")
st.info(
    "Credit card fraud is a growing threat in the digital economy. "
    "Traditional rule-based systems fail to keep pace with evolving fraud tactics. "
    "Machine learning models — optimised for **Recall** — can dramatically improve "
    "detection rates, catching fraudulent transactions that would otherwise slip through."
)

# ---- Team Information ----
st.markdown("### Team Information")
team_data = {
    "Name": ["Samkit Jain", "Mann Makhecha", "Krish Jain", "Chirag Gupta", "Het Malvaniya", "Dhiraj Jagwani"],
}
st.table(team_data)

# ---- Tech Stack ----
st.markdown("### Tech Stack")
t1, t2, t3, t4, t5 = st.columns(5)
t1.markdown("![Python](https://img.icons8.com/color/28/python.png) **Python**")
t2.markdown("![Streamlit](https://img.icons8.com/color/28/streamlit.png) **Streamlit**")
t3.markdown("![Scikit](https://img.icons8.com/color/28/artificial-intelligence.png) **Scikit-learn**")
t4.markdown("![Pandas](https://img.icons8.com/color/28/pandas.png) **Pandas / NumPy**")
t5.markdown("![Chart](https://img.icons8.com/color/28/combo-chart.png) **Matplotlib / Seaborn**")
