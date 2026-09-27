"""
Evaluation utilities shared by train.py and (optionally) ad-hoc analysis:
metric computation, confusion matrix / ROC plotting, classification reports.
"""
import matplotlib
matplotlib.use("Agg")  # safe for headless/script use (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_curve)

from . import config


def evaluate_model(model, generator, steps):
    """Run the model over an entire generator and compute standard metrics."""
    generator.reset()
    y_true = generator.classes
    y_prob = model.predict(generator, steps=steps, verbose=0).ravel()[: len(y_true)]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics["roc_auc"] = float(auc(fpr, tpr))
    return metrics, y_true, y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=config.CLASS_NAMES):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_classification_report(y_true, y_pred, save_path, class_names=config.CLASS_NAMES):
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(save_path, "w") as f:
        f.write(report)
    return report
