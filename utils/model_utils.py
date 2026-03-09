"""
Model training, evaluation, and persistence utilities.
Train-once architecture: all 3 models are trained on first launch,
results cached to disk, and loaded instantly on subsequent runs.
"""

import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Fixed model configs — these are the defaults used for one-time training
MODEL_CONFIGS = {
    "Logistic Regression": {"C": 1.0, "max_iter": 500},
    "Decision Tree": {"max_depth": 10, "min_samples_split": 5},
    "Random Forest": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5},
}


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
    """Train a model and return (fitted_model, training_time_seconds)."""
    model = _build_model(name, params)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a fitted model and return metrics dict + confusion matrix + report."""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    # Precision-Recall curve data
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)

    return metrics, cm, report, y_pred, y_proba, fpr, tpr, pr_precision, pr_recall


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


def _results_path(name: str) -> str:
    safe_name = name.lower().replace(" ", "_")
    return os.path.join(MODELS_DIR, f"{safe_name}_results.pkl")


def save_results(name: str, results: dict):
    """Save evaluation results to disk."""
    joblib.dump(results, _results_path(name))


def load_results(name: str):
    """Load evaluation results from disk."""
    path = _results_path(name)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ---------------------------------------------------------------------------
# Train-once orchestration
# ---------------------------------------------------------------------------
def are_models_trained() -> bool:
    """Check if all 3 models + results are cached on disk."""
    for name in MODEL_CONFIGS:
        safe = name.lower().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe}.pkl")
        results_path = os.path.join(MODELS_DIR, f"{safe}_results.pkl")
        if not os.path.exists(model_path) or not os.path.exists(results_path):
            return False
    return True


def train_all_models(X_train, X_test, y_train, y_test, feature_names, progress_callback=None):
    """
    Train all 3 models, evaluate, and save to disk.
    progress_callback(pct, text) is called if provided (for Streamlit progress bars).
    Returns dict: {model_name: {model, metrics, cm, report, ...}}
    """
    all_results = {}
    model_names = list(MODEL_CONFIGS.keys())

    for i, name in enumerate(model_names):
        params = MODEL_CONFIGS[name]
        pct_base = int((i / len(model_names)) * 100)

        if progress_callback:
            progress_callback(pct_base + 5, f"Training {name}...")

        model, elapsed = train_model(name, X_train, y_train, params)

        if progress_callback:
            progress_callback(pct_base + 20, f"Evaluating {name}...")

        metrics, cm, report, y_pred, y_proba, fpr, tpr, pr_prec, pr_rec = evaluate_model(
            model, X_test, y_test
        )

        # Save model
        save_model(model, name)

        # Build feature importance (for tree-based models)
        feat_importance = None
        if hasattr(model, "feature_importances_"):
            feat_importance = dict(zip(feature_names, model.feature_importances_.tolist()))

        results = {
            "model": model,
            "params": params,
            "metrics": metrics,
            "cm": cm,
            "report": report,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "fpr": fpr,
            "tpr": tpr,
            "pr_precision": pr_prec,
            "pr_recall": pr_rec,
            "elapsed": elapsed,
            "feature_importance": feat_importance,
        }

        # Save results to disk
        save_results(name, results)
        all_results[name] = results

    if progress_callback:
        progress_callback(100, "All models trained!")

    return all_results


def load_all_results():
    """Load all model results from disk. Returns dict or None if any missing."""
    if not are_models_trained():
        return None

    all_results = {}
    for name in MODEL_CONFIGS:
        results = load_results(name)
        model = load_model(name)
        if results is None or model is None:
            return None
        results["model"] = model
        all_results[name] = results

    return all_results
