"""Refusal-threshold selection on DEVELOPMENT data only, exactly as THRESHOLD_SELECTION_METHOD.md (frozen with protocol v2).

Pure functions (numpy/scipy only); no model, no file access. Inputs are per-image records:
  unsupported: dicts with score (top-1 softmax), photographer, group
  supported:   dicts with score, photographer, taxon, correct (top-1 == true taxon)
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
from scipy.stats import beta

TARGET = 0.10
B = 10000
SEED = 20261015
FOLDS = 5
FOLD_SALT = "agrovaidya-m4-d2-folds-v1"
GROUPS = ("N_compound_leaved_neem_like", "M_glossy_simple_leaved", "F_peepal_like", "A_small_leaved_thorny", "P_palms_outside_vocabulary")
MIN = {"unsupported_images": 200, "unsupported_photographers": 100, "group_images": 30, "group_photographers": 15,
       "max_taxon_share": 0.15, "max_photographer_share": 0.05, "supported_images": 100}


def cp_upper(x: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if x >= n else float(beta.ppf(1 - alpha, x + 1, n - x))


class Bound:
    """U(tau) = max(Clopper-Pearson, cluster-bootstrap micro, stratified cluster-bootstrap macro), one-sided 95%."""

    def __init__(self, unsupported: list[dict], seed: int = SEED, b: int = B):
        self.scores = np.array([r["score"] for r in unsupported])
        rng = np.random.default_rng(seed)
        phot = sorted({r["photographer"] for r in unsupported})
        pidx = {p: i for i, p in enumerate(phot)}
        self.c_of = np.array([pidx[r["photographer"]] for r in unsupported])
        self.n_c = np.bincount(self.c_of, minlength=len(phot)).astype(float)
        self.W = rng.multinomial(len(phot), np.full(len(phot), 1 / len(phot)), size=b).astype(float)      # B x C
        self.groups = []
        for g in GROUPS:
            m = np.array([r["group"] == g for r in unsupported])
            if not m.any():
                continue
            gp = sorted({r["photographer"] for r, k in zip(unsupported, m) if k})
            gidx = {p: i for i, p in enumerate(gp)}
            c = np.array([gidx[r["photographer"]] for r, k in zip(unsupported, m) if k])
            Wg = rng.multinomial(len(gp), np.full(len(gp), 1 / len(gp)), size=b).astype(float)
            self.groups.append((m, c, np.bincount(c, minlength=len(gp)).astype(float), Wg))

    def components(self, tau: float) -> dict:
        acc = self.scores >= tau
        x, n = int(acc.sum()), len(acc)
        acc_c = np.bincount(self.c_of, weights=acc.astype(float), minlength=len(self.n_c))
        micro = np.percentile((self.W @ acc_c) / (self.W @ self.n_c), 95)
        fa_g = []
        for m, c, n_gc, Wg in self.groups:
            a_gc = np.bincount(c, weights=acc[m].astype(float), minlength=len(n_gc))
            fa_g.append((Wg @ a_gc) / (Wg @ n_gc))
        macro = np.percentile(np.mean(fa_g, axis=0), 95) if fa_g else 1.0
        return {"accepted": x, "n": n, "fa": x / n, "cp_upper": cp_upper(x, n), "boot_micro_upper": float(micro),
                "boot_macro_upper": float(macro), "U": max(cp_upper(x, n), float(micro), float(macro))}


def candidates(unsupported: list[dict]) -> list[float]:
    s = sorted({r["score"] for r in unsupported})
    return [min(s)] + [v + 1e-9 for v in s]


def select(unsupported: list[dict], seed: int = SEED, b: int = B) -> dict:
    """Smallest candidate tau with U(tau) <= TARGET (U is non-increasing in tau, so binary search is exact)."""
    bound = Bound(unsupported, seed, b)
    cand = candidates(unsupported)
    lo, hi = 0, len(cand) - 1
    if bound.components(cand[hi])["U"] > TARGET:
        return {"tau": None, "reason": "no candidate satisfies the bound (too few unsupported images)"}
    while lo < hi:
        mid = (lo + hi) // 2
        if bound.components(cand[mid])["U"] <= TARGET:
            hi = mid
        else:
            lo = mid + 1
    return {"tau": cand[lo], **bound.components(cand[lo])}


def supported_metrics(supported: list[dict], tau: float) -> dict:
    n = len(supported)
    ans = [r for r in supported if r["score"] >= tau]
    corr = sum(r["correct"] for r in ans)
    return {"n": n, "coverage": len(ans) / n if n else 0.0, "correct_with_refusals": corr / n if n else 0.0,
            "selective_accuracy": corr / len(ans) if ans else None}


def minimum_checks(unsupported: list[dict], supported: list[dict]) -> dict:
    out = {}
    n = len(unsupported)
    ph = Counter(r["photographer"] for r in unsupported)
    tx = Counter(r["taxon"] for r in unsupported)
    out["unsupported_images"] = (n, n >= MIN["unsupported_images"])
    out["unsupported_photographers"] = (len(ph), len(ph) >= MIN["unsupported_photographers"])
    for g in GROUPS:
        rs = [r for r in unsupported if r["group"] == g]
        k = len({r["photographer"] for r in rs})
        out[f"group_{g}"] = ((len(rs), k), len(rs) >= MIN["group_images"] and k >= MIN["group_photographers"])
    out["max_taxon_share"] = (round(max(tx.values()) / n, 4) if n else 1, bool(n) and max(tx.values()) / n <= MIN["max_taxon_share"])
    out["max_photographer_share"] = (round(max(ph.values()) / n, 4) if n else 1,
                                     bool(n) and max(ph.values()) / n <= MIN["max_photographer_share"])
    out["supported_images"] = (len(supported), len(supported) >= MIN["supported_images"])
    return out


def fold_of(photographer: str) -> int:
    return int(hashlib.sha256(f"{FOLD_SALT}|{photographer}".encode()).hexdigest(), 16) % FOLDS


def cross_fit(unsupported: list[dict], supported: list[dict], seed: int = SEED, b: int = B) -> list[dict]:
    out = []
    for k in range(FOLDS):
        tr = [r for r in unsupported if fold_of(r["photographer"]) != k]
        te = [r for r in unsupported if fold_of(r["photographer"]) == k]
        sel = select(tr, seed, b)
        tau = sel["tau"]
        st = [r for r in supported if fold_of(r["photographer"]) == k]
        row = {"fold": k, "tau": tau, "heldout_unsupported": len(te), "heldout_supported": len(st)}
        if tau is not None:
            row["heldout_fa"] = sum(r["score"] >= tau for r in te) / len(te) if te else None
            row.update({f"heldout_{a}": v for a, v in supported_metrics(st, tau).items() if a != "n"})
        out.append(row)
    return out


def run(unsupported: list[dict], supported: list[dict], seed: int = SEED, b: int = B) -> dict:
    mins = minimum_checks(unsupported, supported)
    failed = [k for k, (_, ok) in mins.items() if not ok]
    if failed:
        return {"status": "unresolved", "reason": f"development minimums not met: {failed}", "minimums": mins}
    sel = select(unsupported, seed, b)
    if sel["tau"] is None:
        return {"status": "unresolved", "reason": sel["reason"], "minimums": mins}
    sup = supported_metrics(supported, sel["tau"])
    folds = cross_fit(unsupported, supported, seed, b)
    taus = [f["tau"] for f in folds if f["tau"] is not None]
    over = sum(1 for f in folds if f.get("heldout_fa") is not None and f["heldout_fa"] > TARGET)
    problems = []
    if sup["coverage"] < 0.70:
        problems.append(f"supported coverage {sup['coverage']:.3f} < 0.70")
    if sup["correct_with_refusals"] < 0.80:
        problems.append(f"supported correct-with-refusals {sup['correct_with_refusals']:.3f} < 0.80")
    if len(taus) < FOLDS:
        problems.append("a fold could not select a threshold")
    elif max(taus) - min(taus) > 0.05:
        problems.append(f"unstable: fold tau range {max(taus) - min(taus):.3f} > 0.05")
    if over > 1:
        problems.append(f"unstable: out-of-fold false acceptance > 10% in {over} folds")
    return {"status": "resolved" if not problems else "unresolved", "reason": "; ".join(problems) or None,
            "tau": sel["tau"], "selection": sel, "supported": sup, "folds": folds, "minimums": mins}
