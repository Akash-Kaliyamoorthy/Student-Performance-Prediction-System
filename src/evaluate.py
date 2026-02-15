"""Evaluate the saved best model and generate reports/plots."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from data_loader import load_dataset
from preprocessing import create_features_and_target, split_data


MODEL_PATH = Path("models/best_model.pkl")
REPORT_PATH = Path("evaluation_report.json")
PLOTS_DIR = Path("plots")


def evaluate_model(random_state: int = 42) -> dict:
    """Load saved model and compute evaluation metrics on test split."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]

    df = load_dataset()
    X, y = create_features_and_target(df)
    split = split_data(X, y, random_state=random_state)

    y_pred = model.predict(split.X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(split.X_test)[:, 1]
    else:
        y_score = model.decision_function(split.X_test)

    accuracy = accuracy_score(split.y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        split.y_test, y_pred, average="binary"
    )
    cm = confusion_matrix(split.y_test, y_pred)

    fpr, tpr, _ = roc_curve(split.y_test, y_score)
    roc_auc = roc_auc_score(split.y_test, y_score)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Student Performance Classifier")
    plt.legend(loc="lower right")
    roc_plot_path = PLOTS_DIR / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    plt.close()

    metrics = {
        "model_name": artifact.get("best_model_name"),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "roc_curve_plot": str(roc_plot_path),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1 Score:", metrics["f1_score"])
    print("Confusion Matrix:", metrics["confusion_matrix"])
    print("ROC Curve plot saved to:", metrics["roc_curve_plot"])

    return metrics


if __name__ == "__main__":
    evaluate_model()
