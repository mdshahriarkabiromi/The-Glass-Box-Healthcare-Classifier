from typing import Dict, Any
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    brier_score_loss
)
from sklearn.calibration import CalibrationDisplay

def evaluate_binary(y_true, y_proba, threshold: float) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist()
    }

def save_diagnostics_plots(y_true, y_proba, out_dir: str, threshold: float) -> None:
    os.makedirs(out_dir, exist_ok=True)

    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve")
    plt.savefig(f"{out_dir}/roc_curve.png", bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{out_dir}/pr_curve.png", bbox_inches="tight")
    plt.close()

    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=10)
    plt.title("Calibration Curve")
    plt.savefig(f"{out_dir}/calibration_curve.png", bbox_inches="tight")
    plt.close()

    y_pred = (y_proba >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.savefig(f"{out_dir}/confusion_matrix.png", bbox_inches="tight")
    plt.close()