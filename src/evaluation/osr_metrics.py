"""Open-set evaluation metrics.

Conventions (used everywhere in this project):
  * score `u` = an UNKNOWN-ness score: higher = more likely unknown. For MSP, u = 1 - MSP,
    computed in float64 without cancellation (see src/osr/scores.py) so that saturated
    confidences (MSP ~ 1) are still ranked instead of tied.
  * A sample is ACCEPTED as known iff u <= t  (equivalently MSP >= 1 - t), else REJECTED as unknown.
  * AUROC and AUPR-Out treat UNKNOWN as the positive class and use u as the score.
    AUROC = 0.5 is chance; < 0.5 would mean the score is inverted.
  * AUPR-In treats KNOWN as positive with score -u.
  * FPR@95TPR: TPR is the known (in-distribution) acceptance rate. Pick the threshold that accepts
    95% of KNOWN samples; FPR is the fraction of UNKNOWN samples accepted at that threshold
    (the usual OOD-literature definition, e.g. Hendrycks & Gimpel 2017; Liang et al. 2018).
  * OSCR (Dhamija et al., 2018): area under CCR(t) vs FPR(t) where
    CCR(t) = #(known, correctly classified, accepted) / #known and FPR(t) = #(unknown accepted) / #unknown.
    The curve runs from (0,0) to (1, closed-set accuracy), so OSCR <= closed-set accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def ranking_metrics(u_known: np.ndarray, u_unknown: np.ndarray) -> dict:
    y = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]          # 1 = unknown
    s = np.r_[u_known, u_unknown]
    fpr_k, tpr_k, _ = roc_curve(1 - y, -s)                               # known positive, score -u
    i = int(np.searchsorted(tpr_k, 0.95, side="left"))
    return {"auroc": float(roc_auc_score(y, s)),
            "aupr_out": float(average_precision_score(y, s)),
            "aupr_in": float(average_precision_score(1 - y, -s)),
            "fpr_at_95tpr": float(fpr_k[min(i, len(fpr_k) - 1)]),
            "n_known": int(len(u_known)), "n_unknown": int(len(u_unknown)),
            "unknown_fraction": float(len(u_unknown) / len(s))}


def oscr(u_known: np.ndarray, correct_known: np.ndarray, u_unknown: np.ndarray) -> float:
    """Exact area under the CCR-vs-FPR curve over all thresholds (accept iff u <= t)."""
    s = np.r_[u_known, u_unknown]
    is_kc = np.r_[correct_known.astype(float), np.zeros(len(u_unknown))]
    is_u = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]
    order = np.argsort(s, kind="mergesort")                                # most "known" first
    s, is_kc, is_u = s[order], is_kc[order], is_u[order]
    ccr = np.cumsum(is_kc) / len(u_known)
    fpr = np.cumsum(is_u) / len(u_unknown)
    last = np.r_[s[1:] != s[:-1], True]                                    # thresholds between distinct scores
    ccr, fpr = np.r_[0.0, ccr[last]], np.r_[0.0, fpr[last]]
    return float(np.trapz(ccr, fpr))


def threshold_metrics(u_known, pred_known, y_known, u_unknown, t: float, n_classes: int) -> dict:
    """Decision metrics at a fixed threshold t (accept iff u <= t)."""
    acc_k = u_known <= t
    acc_u = u_unknown <= t
    correct = pred_known == y_known
    n_k, n_u = len(u_known), len(u_unknown)
    rej_u, rej_k = int((~acc_u).sum()), int((~acc_k).sum())
    rec = rej_u / n_u if n_u else float("nan")
    prec = rej_u / (rej_u + rej_k) if (rej_u + rej_k) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if n_u and (prec + rec) > 0 else float("nan")
    return {
        "known_closed_set_accuracy": float(correct.mean()),
        "known_acceptance_rate": float(acc_k.mean()),
        "false_rejection_rate": float(1 - acc_k.mean()),
        "known_accuracy_among_accepted": float(correct[acc_k].mean()) if acc_k.any() else float("nan"),
        "known_correct_and_accepted_rate": float((correct & acc_k).mean()),      # = CCR(t)
        "unknown_recall": float(rec), "unknown_precision": float(prec), "unknown_f1": float(f1),
        "false_known_rate": float(1 - rec) if n_u else float("nan"),
        "open_set_accuracy": float(((correct & acc_k).sum() + rej_u) / (n_k + n_u)),
        "rejected_known": rej_k, "rejected_unknown": rej_u, "n_known": n_k, "n_unknown": n_u,
    }


def open_set_macro_f1(pred_known, acc_known, y_known, pred_unknown, acc_unknown, n_classes: int) -> float:
    """Macro-F1 over n_classes known labels + 1 'unknown' label (accepted unknowns keep their argmax class)."""
    true = np.r_[y_known, np.full(len(pred_unknown), n_classes)]
    pred = np.r_[np.where(acc_known, pred_known, n_classes), np.where(acc_unknown, pred_unknown, n_classes)]
    return float(f1_score(true, pred, labels=list(range(n_classes + 1)), average="macro", zero_division=0))


def per_unknown_class(u_known, df_unknown, t: float) -> list[dict]:
    """Per unknown class: recall = rejected share; precision/F1 as if the pool were known + this class only."""
    rej_k = int((u_known > t).sum())
    out = []
    for c, d in df_unknown.groupby("class_name"):
        u = d["u"].to_numpy()
        rej = int((u > t).sum())
        rec = rej / len(u)
        prec = rej / (rej + rej_k) if (rej + rej_k) else float("nan")
        out.append({"class_name": c, "crop": d["crop"].iat[0], "dataset": d["dataset"].iat[0], "images": int(len(u)),
                    "unknown_recall": rec, "unknown_precision": prec,
                    "unknown_f1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
                    "mean_msp": float(d["msp"].mean()) if "msp" in d else float("nan"),
                    "median_msp": float(d["msp"].median()) if "msp" in d else float("nan"),
                    "auroc_vs_known": float(roc_auc_score(np.r_[np.zeros(len(u_known)), np.ones(len(u))],
                                                          np.r_[u_known, u]))})
    return out
