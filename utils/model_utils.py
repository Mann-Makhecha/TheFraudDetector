"""
Model training, evaluation, and persistence utilities.
"""

import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _build_model(name: str, params: dict):
    """Return an sklearn estimator from a name string + param dict."""
    if name == "Logistic Regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 500),
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )
    elif name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 5),
            class_weight="balanced",
            random_state=42,
        )
    elif name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 15),
            min_samples_split=params.get("min_samples_split", 5),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------
def train_model(name: str, X_train, y_train, params: dict):
    """
    Train a model and return (fitted_model, training_time_seconds).
    """
    model = _build_model(name, params)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a fitted model and return a metrics dict + confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    metrics = {
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
    return metrics, cm, report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model, name: str):
    """Persist a trained model to disk."""
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """Load a previously saved model."""
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None
