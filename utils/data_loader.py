"""
Data loading, feature engineering, and preprocessing utilities.
All heavy I/O is cached with @st.cache_data for performance.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fraudTrain.csv")
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fraudTest.csv")

# Columns to drop (PII / high-cardinality identifiers)
DROP_COLS = ["Unnamed: 0", "cc_num", "first", "last", "street", "trans_num", "merchant"]

# Categories present in the training data (fixed order for one-hot encoding)
CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_net",
    "shopping_pos", "travel",
]

GENDERS = ["F", "M"]


# ---------------------------------------------------------------------------
# 1. Load raw data (cached to Parquet on disk for fast restarts)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading training dataset (checking cache)…")
def load_data() -> pd.DataFrame:
    """Load fraudTrain.csv with optimised dtypes and parsed dates."""
    parquet_path = DATA_PATH.replace('.csv', '.parquet')
    
    # Fast load from 0.5s local cache
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    dtype_map = {
        "category": "str",
        "amt": "float32",
        "gender": "str",
        "city": "str",
        "state": "str",
        "zip": "int32",
        "lat": "float32",
        "long": "float32",
        "city_pop": "int32",
        "job": "str",
        "merch_lat": "float32",
        "merch_long": "float32",
        "is_fraud": "int8",
    }
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["trans_date_trans_time", "dob"],
        dtype=dtype_map,
        low_memory=False,
    )
    # Drop PII / useless columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")
    
    # Save to disk for next time
    df.to_parquet(parquet_path, engine='pyarrow')
    return df


@st.cache_data(show_spinner="Loading test dataset (checking cache)…")
def load_test_data() -> pd.DataFrame:
    """Load fraudTest.csv with optimised dtypes and parsed dates."""
    parquet_path = TEST_DATA_PATH.replace('.csv', '.parquet')
    
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    dtype_map = {
        "category": "str",
        "amt": "float32",
        "gender": "str",
        "city": "str",
        "state": "str",
        "zip": "int32",
        "lat": "float32",
        "long": "float32",
        "city_pop": "int32",
        "job": "str",
        "merch_lat": "float32",
        "merch_long": "float32",
        "is_fraud": "int8",
    }
    df = pd.read_csv(
        TEST_DATA_PATH,
        parse_dates=["trans_date_trans_time", "dob"],
        dtype=dtype_map,
        low_memory=False,
    )
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")
    
    df.to_parquet(parquet_path, engine='pyarrow')
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used for EDA and modelling."""
    df = df.copy()
    dt = df["trans_date_trans_time"]
    df["hour"] = dt.dt.hour.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")  # 0=Mon
    df["month"] = dt.dt.month.astype("int8")
    # Age in years at time of transaction
    df["age"] = (
        (dt - df["dob"]).dt.days / 365.25
    ).astype("float32")
    # Distance between cardholder and merchant
    df["distance_km"] = _haversine(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    ).astype("float32")
    # Log-transformed amount
    df["amt_log"] = np.log1p(df["amt"]).astype("float32")
    return df


# ---------------------------------------------------------------------------
# 3. High-Performance Zero-Arg Wrappers
# ---------------------------------------------------------------------------
# Passing a 1.5GB DataFrame to @st.cache_data causes Streamlit to spend 
# several seconds hashing it. These zero-arg wrappers prevent Streamlit 
# from hashing the data on page loads, making page switching instant.

@st.cache_data(show_spinner="Engineering features…", ttl=86400)
def get_engineered_train() -> pd.DataFrame:
    return engineer_features(load_data())

@st.cache_data(show_spinner="Engineering features…", ttl=86400)
def get_engineered_test() -> pd.DataFrame:
    return engineer_features(load_test_data())


# ---------------------------------------------------------------------------
# 4. Model-ready features
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "amt", "amt_log", "hour", "day_of_week", "month",
    "age", "city_pop", "distance_km",
]

@st.cache_data(show_spinner="Preparing model features…", ttl=86400)
def get_model_features(df: pd.DataFrame):
    """
    Return (X, y, feature_names).
    """
    X = df[NUMERIC_FEATURES].copy()

    # One-hot: category
    for cat in CATEGORIES:
        X[f"category_{cat}"] = (df["category"] == cat).astype("int8")

    # One-hot: gender
    for g in GENDERS:
        X[f"gender_{g}"] = (df["gender"] == g).astype("int8")

    y = df["is_fraud"].values
    feature_names = list(X.columns)
    return X, y, feature_names


def build_single_row(amt, category, hour, age,
                     day_of_week=2, month=6, city_pop=50000,
                     distance_km=30.0, gender="F"):
    """Build a single-row DataFrame matching the training feature set."""
    row = {feat: 0.0 for feat in NUMERIC_FEATURES}
    row["amt"] = amt
    row["amt_log"] = float(np.log1p(amt))
    row["hour"] = hour
    row["day_of_week"] = day_of_week
    row["month"] = month
    row["age"] = age
    row["city_pop"] = city_pop
    row["distance_km"] = distance_km

    for cat in CATEGORIES:
        row[f"category_{cat}"] = 1 if cat == category else 0
    for g in GENDERS:
        row[f"gender_{g}"] = 1 if g == gender else 0

    feature_names = (
        NUMERIC_FEATURES
        + [f"category_{c}" for c in CATEGORIES]
        + [f"gender_{g}" for g in GENDERS]
    )
    return pd.DataFrame([row], columns=feature_names)
