"""Closed-set classification metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f), "weighted_f1": float(fw)}


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class_id": labels, "class_name": class_names, "precision": p, "recall": r,
                         "f1": f, "support": s})


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def top_confusions(cm: np.ndarray, class_names: list[str], k: int = 10) -> pd.DataFrame:
    """Largest off-diagonal counts, with the share of the true class they represent."""
    rows = []
    for t in range(len(cm)):
        for p in range(len(cm)):
            if t != p and cm[t, p] > 0:
                rows.append({"true": class_names[t], "predicted": class_names[p], "count": int(cm[t, p]),
                             "share_of_true_class": float(cm[t, p] / cm[t].sum())})
    df = pd.DataFrame(rows, columns=["true", "predicted", "count", "share_of_true_class"])
    return df.sort_values(["count", "share_of_true_class"], ascending=False).head(k).reset_index(drop=True)
