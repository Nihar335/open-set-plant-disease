"""Sealed-test analysis for protocol v2 section 7 (pure functions; used only by evaluate_d2.py --phase sealed, which
requires the owner's hash-bound authorisation). Unit-tested on synthetic data so the authorised run is reproducible.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import beta

from stage10.plant_id.d2.thresholds import B, SEED, TARGET, cp_upper


def cp_two_sided(x: int, n: int) -> tuple[float, float]:
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def cluster_bootstrap(values: list[float], clusters: list[str], seed: int = SEED, b: int = B) -> np.ndarray:
    """Means of `values` over photographer-cluster bootstrap resamples."""
    keys = sorted(set(clusters))
    idx = {k: i for i, k in enumerate(keys)}
    c = np.array([idx[k] for k in clusters])
    s = np.bincount(c, weights=np.asarray(values, float), minlength=len(keys))
    n = np.bincount(c, minlength=len(keys)).astype(float)
    W = np.random.default_rng(seed).multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=b).astype(float)
    return (W @ s) / (W @ n)


def unsupported_outcome(rows: list[dict], tau: float) -> dict:
    acc = [float(r["score"] >= tau) for r in rows]
    x, n = int(sum(acc)), len(acc)
    boot = cluster_bootstrap(acc, [r["photographer"] for r in rows])
    upper = max(cp_upper(x, n), float(np.percentile(boot, 95)))
    verdict = "met in this study" if upper <= TARGET else ("not met" if x / n > TARGET else "not established")
    wrong = Counter(r["top1"] for r, a in zip(rows, acc) if a)
    return {"n": n, "accepted": x, "false_acceptance": x / n, "one_sided_95_upper": upper,
            "two_sided_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "cp_two_sided": cp_two_sided(x, n),
            "verdict": verdict, "wrongly_accepted_as": wrong.most_common()}


def supported_outcome(rows: list[dict], tau: float) -> dict:
    ans = [float(r["score"] >= tau) for r in rows]
    cwr = [float(r["score"] >= tau and r["top1"] == r["taxon"]) for r in rows]
    ph = [r["photographer"] for r in rows]
    bc = cluster_bootstrap(cwr, ph)
    n = len(rows)
    cov, acc = sum(ans) / n, sum(cwr) / n
    lower = float(np.percentile(bc, 2.5))
    verdict = ("met in this study" if lower >= 0.80 and cov >= 0.70 else
               "not met" if acc < 0.80 or cov < 0.70 else "not established")
    return {"n": n, "coverage": cov, "correct_with_refusals": acc, "cwr_95": [lower, float(np.percentile(bc, 97.5))],
            "selective_accuracy": sum(cwr) / sum(ans) if sum(ans) else None,
            "top5": sum(r["taxon"] in r["top5"] for r in rows) / n, "verdict": verdict}


def c1_outcome(rows: list[dict], tau: float) -> dict:
    out = {}
    for true, other in (("Cocos nucifera", "Areca catechu"), ("Areca catechu", "Cocos nucifera")):
        rs = [r for r in rows if r["taxon"] == true]
        conf_all = sum(r["top1"] == other for r in rs)
        conf_ans = sum(r["top1"] == other and r["score"] >= tau for r in rs)
        d = {"n": len(rs), "confused_top1": conf_all, "confused_answered": conf_ans,
             "answered": sum(r["score"] >= tau for r in rs)}
        if true == "Cocos nucifera":
            d["verdict"] = ("met in this study" if len(rs) >= 60 and conf_ans <= 1 else
                            "not met" if conf_ans > 1 else "not established (n < 60)")
        else:
            d["verdict"] = "descriptive only (areca below planned sample size)"
        out[f"{true} -> {other}"] = d
    return out


def by_key(rows: list[dict], key: str, tau: float) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    out = {}
    for k, rs in sorted(g.items()):
        a = sum(r["score"] >= tau for r in rs)
        d = {"n": len(rs), "answered_or_accepted": a}
        if len(rs) >= 10:
            d["rate"] = a / len(rs)
            if len(rs) < 30:
                d["warning"] = "n < 30: wide interval"
        out[k] = d
    return out
