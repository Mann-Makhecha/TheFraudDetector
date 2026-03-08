"""
Reusable visualisation functions for the EDA page.
Each function returns a matplotlib Figure so the caller can do st.pyplot(fig).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", palette="muted")

# Consistent colour palette for fraud labels
FRAUD_PALETTE = {0: "#2ecc71", 1: "#e74c3c"}
FRAUD_LABELS = {0: "Legitimate", 1: "Fraud"}

import streamlit as st


@st.cache_data(show_spinner=False)
def plot_fraud_countplot(df: pd.DataFrame):
    """1. Countplot — Fraud vs. Non-fraud distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["is_fraud"].value_counts().sort_index()
    bars = ax.bar(
        [FRAUD_LABELS[i] for i in counts.index],
        counts.values,
        color=[FRAUD_PALETTE[i] for i in counts.index],
        edgecolor="white",
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Transaction Distribution: Fraud vs. Legitimate")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def plot_fraud_rate_by_category(df: pd.DataFrame):
    """2. Bar chart — Fraud rate (%) per merchant category."""
    fraud_rate = (
        df.groupby("category")["is_fraud"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fraud_rate["is_fraud"] *= 100  # percentage

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=fraud_rate, x="is_fraud", y="category", ax=ax,
                palette="Reds_r", edgecolor="white")
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_ylabel("")
    ax.set_title("Fraud Rate by Merchant Category")
    for i, (val, cat) in enumerate(zip(fraud_rate["is_fraud"], fraud_rate["category"])):
        ax.text(val + 0.05, i, f"{val:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def plot_amount_histogram(df: pd.DataFrame):
    """3. Histogram — Transaction amount for Fraud vs. Legitimate."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in [0, 1]:
        subset = df[df["is_fraud"] == label]["amt"]
        ax.hist(subset, bins=80, alpha=0.65, label=FRAUD_LABELS[label],
                color=FRAUD_PALETTE[label], edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Transaction Amount ($) — log scale")
    ax.set_ylabel("Frequency")
    ax.set_title("Transaction Amount Distribution: Fraud vs. Legitimate")
    ax.legend()
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def plot_amount_boxplot(df: pd.DataFrame):
    """4. Box plot — Transaction amount by fraud label."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = df[["amt", "is_fraud"]].copy()
    plot_df["Label"] = plot_df["is_fraud"].map(FRAUD_LABELS)
    sns.boxplot(data=plot_df, x="Label", y="amt", ax=ax,
                palette=FRAUD_PALETTE.values(), showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Transaction Amount ($) — log scale")
    ax.set_xlabel("")
    ax.set_title("Transaction Amount by Fraud Label")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def plot_fraud_heatmap(df: pd.DataFrame):
    """5. Heatmap — Fraud count by Hour of Day vs. Day of Week."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fraud_only = df[df["is_fraud"] == 1]
    pivot = fraud_only.pivot_table(
        index="hour", columns="day_of_week", values="is_fraud",
        aggfunc="count", fill_value=0,
    )
    # Ensure all days/hours present
    pivot = pivot.reindex(index=range(24), columns=range(7), fill_value=0)
    pivot.columns = day_names

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, ax=ax,
                fmt="d", annot=True, annot_kws={"size": 7})
    ax.set_ylabel("Hour of Day")
    ax.set_xlabel("Day of Week")
    ax.set_title("Fraud Transactions: Hour of Day vs. Day of Week")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def plot_top_states(df: pd.DataFrame, top_n: int = 15):
    """6. Bar chart — Top states by fraud count."""
    fraud_only = df[df["is_fraud"] == 1]
    state_counts = fraud_only["state"].value_counts().head(top_n).reset_index()
    state_counts.columns = ["state", "fraud_count"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=state_counts, x="fraud_count", y="state", ax=ax,
                palette="flare", edgecolor="white")
    ax.set_xlabel("Fraud Count")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} States by Fraud Count")
    for i, val in enumerate(state_counts["fraud_count"]):
        ax.text(val + 1, i, f"{val:,}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def plot_fraud_over_time(df: pd.DataFrame):
    """7. Line chart — Monthly fraud transaction volume over time."""
    fraud_only = df[df["is_fraud"] == 1].copy()
    fraud_only["year_month"] = fraud_only["trans_date_trans_time"].dt.to_period("M")
    monthly = fraud_only.groupby("year_month").size().reset_index(name="fraud_count")
    monthly["year_month"] = monthly["year_month"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["year_month"], monthly["fraud_count"],
            marker="o", color="#e74c3c", linewidth=2, markersize=6)
    ax.fill_between(monthly["year_month"], monthly["fraud_count"],
                    alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Month")
    ax.set_ylabel("Fraud Transactions")
    ax.set_title("Fraud Transaction Volume Over Time")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels=None):
    """Confusion matrix heatmap for the model training page."""
    if labels is None:
        labels = ["Legitimate", "Fraud"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics_dict: dict):
    """
    Grouped bar chart comparing models on Precision, Recall, F1, ROC-AUC.
    metrics_dict: {model_name: {metric_name: value, ...}, ...}
    """
    metric_names = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    n_models = len(model_names)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(model_names):
        vals = [metrics_dict[model].get(m, 0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors[i % len(colors)], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Precision · Recall · F1 · ROC-AUC")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
