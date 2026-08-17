from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score


def evaluate_binary(y_true, probabilities, threshold: float = 0.5) -> dict[str, Any]:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"threshold": threshold, "accuracy": accuracy_score(y_true, predicted), "precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0), "specificity": tn / (tn + fp) if tn + fp else 0.0, "f1": f1_score(y_true, predicted, zero_division=0), "roc_auc": roc_auc_score(y_true, probabilities), "pr_auc": average_precision_score(y_true, probabilities), "log_loss": log_loss(y_true, probabilities), "brier_score": brier_score_loss(y_true, probabilities), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
