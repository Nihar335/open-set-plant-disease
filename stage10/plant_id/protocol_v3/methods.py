"""Protocol v3 candidate decision rules and development-only threshold selection (PROTOCOL_V3_PROPOSAL.md section 4).

Pure functions (numpy/scipy only); no model, no file access. The v2 bound (stage10.plant_id.d2.thresholds.Bound) is reused
unchanged, with the same seed, B and bootstrap draws, so the false-acceptance criterion is identical to protocol v2.

Records: dicts with probs (list of 40 floats in vocabulary order), taxon, role, group, photographer.
"""
from __future__ import annotations

import numpy as np

from stage10.plant_id.d2 import thresholds as T

COCONUT = "Cocos nucifera"
ARECA = "Areca catechu"
PAIR = (COCONUT, ARECA)
SIBLING = {COCONUT: ARECA, ARECA: COCONUT}
CANDIDATES = ("A", "B", "C")
REFUSED = -1.0                      # score given to a record that a rule never answers (below every probability)


def top1(rec: dict, names: list[str]) -> tuple[str, float]:
    p = rec["probs"]
    i = int(np.argmax(p))
    return names[i], float(p[i])


def score_v2(rec: dict, names: list[str]) -> float:
    return top1(rec, names)[1]


def score_a(rec: dict, names: list[str]) -> float:
    """V3-A: a palm top-1 is never answered ('palm: cannot identify; please confirm the crop')."""
    t, s = top1(rec, names)
    return REFUSED if t in PAIR else s


def score_c(rec: dict, names: list[str]) -> float:
    """V3-C: for a palm top-1, the top-1 probability renormalised after removing its vocabulary sibling."""
    t, s = top1(rec, names)
    if t not in PAIR:
        return s
    sib = rec["probs"][names.index(SIBLING[t])]
    return s / (1.0 - sib) if sib < 1.0 else s


def _records(recs: list[dict], names: list[str], scorer) -> tuple[list[dict], list[dict]]:
    uns, sup = [], []
    for r in recs:
        t, _ = top1(r, names)
        row = {"score": scorer(r, names), "photographer": r["photographer"], "group": r["group"], "taxon": r["taxon"], "top1": t}
        if r["role"] == "unsupported":
            uns.append(row)
        else:
            sup.append({**row, "correct": t == r["taxon"]})
    return uns, sup


def run_single(recs: list[dict], names: list[str], candidate: str) -> dict:
    """V3-A and V3-C: one global tau on a transformed score, selected by the v2 method exactly (thresholds.run)."""
    scorer = {"A": score_a, "C": score_c}[candidate]
    uns, sup = _records(recs, names, scorer)
    res = T.run(uns, sup)
    res["candidate"] = candidate
    if res.get("tau") is not None:
        res["supported_by_taxon"] = by_taxon(sup, lambda r: r["score"] >= res["tau"])
        res["unsupported_by_group"] = by_group(uns, lambda r: r["score"] >= res["tau"])
    return res


# ------------------------------------------------------------------------------------------------- V3-B (two strata)
def _acc_components(bound: T.Bound, acc: np.ndarray) -> dict:
    """The v2 U, computed for an arbitrary acceptance vector with the bound's own bootstrap draws."""
    x, n = int(acc.sum()), len(acc)
    acc_c = np.bincount(bound.c_of, weights=acc.astype(float), minlength=len(bound.n_c))
    micro = float(np.percentile((bound.W @ acc_c) / (bound.W @ bound.n_c), 95))
    fa_g = []
    for m, c, n_gc, Wg in bound.groups:
        a_gc = np.bincount(c, weights=acc[m].astype(float), minlength=len(n_gc))
        fa_g.append((Wg @ a_gc) / (Wg @ n_gc))
    macro = float(np.percentile(np.mean(fa_g, axis=0), 95)) if fa_g else 1.0
    cp = T.cp_upper(x, n)
    return {"accepted": x, "n": n, "fa": x / n, "cp_upper": cp, "boot_micro_upper": micro, "boot_macro_upper": macro,
            "U": max(cp, micro, macro)}


def accept_b(top1_name: str, score: float, tau_g: float, tau_c: float) -> bool:
    return score >= (tau_c if top1_name == COCONUT else tau_g)


def _cands(scores: list[float]) -> list[float]:
    s = sorted(set(scores))
    return [min(s)] + [v + 1e-9 for v in s]


def _smallest(cands: list[float], ok) -> float | None:
    if not ok(cands[-1]):
        return None
    lo, hi = 0, len(cands) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(cands[mid]):
            hi = mid
        else:
            lo = mid + 1
    return cands[lo]


def select_b(uns: list[dict], seed: int = T.SEED, b: int = T.B) -> dict:
    bound = T.Bound(uns, seed, b)
    s = np.array([r["score"] for r in uns])
    is_c = np.array([r["top1"] == COCONUT for r in uns])
    gen_scores = [r["score"] for r in uns if r["top1"] != COCONUT]
    if not gen_scores:
        return {"tau_g": None, "tau_c": None, "reason": "no general-stratum unsupported image"}
    tau_g = _smallest(_cands(gen_scores), lambda t: _acc_components(bound, (~is_c) & (s >= t))["U"] <= T.TARGET)
    if tau_g is None:
        return {"tau_g": None, "tau_c": None, "reason": "no general threshold satisfies the bound"}
    c_scores = [r["score"] for r in uns if r["top1"] == COCONUT]
    if not c_scores:
        tau_c = tau_g
    else:
        tau_c = _smallest(_cands(c_scores),
                          lambda t: _acc_components(bound, ((~is_c) & (s >= tau_g)) | (is_c & (s >= t)))["U"] <= T.TARGET)
    acc = ((~is_c) & (s >= tau_g)) | (is_c & (s >= tau_c))
    return {"tau_g": tau_g, "tau_c": tau_c, **_acc_components(bound, acc)}


def _sup_metrics_b(sup: list[dict], tau_g: float, tau_c: float) -> dict:
    n = len(sup)
    ans = [r for r in sup if accept_b(r["top1"], r["score"], tau_g, tau_c)]
    corr = sum(r["correct"] for r in ans)
    return {"n": n, "coverage": len(ans) / n if n else 0.0, "correct_with_refusals": corr / n if n else 0.0,
            "selective_accuracy": corr / len(ans) if ans else None}


def run_b(recs: list[dict], names: list[str], seed: int = T.SEED, b: int = T.B) -> dict:
    uns, sup = _records(recs, names, score_v2)
    mins = T.minimum_checks(uns, sup)
    failed = [k for k, (_, ok) in mins.items() if not ok]
    if failed:
        return {"candidate": "B", "status": "unresolved", "reason": f"development minimums not met: {failed}", "minimums": mins}
    sel = select_b(uns, seed, b)
    if sel["tau_g"] is None:
        return {"candidate": "B", "status": "unresolved", "reason": sel["reason"], "minimums": mins}
    tg, tc = sel["tau_g"], sel["tau_c"]
    supm = _sup_metrics_b(sup, tg, tc)
    folds = []
    for k in range(T.FOLDS):
        tr = [r for r in uns if T.fold_of(r["photographer"]) != k]
        te = [r for r in uns if T.fold_of(r["photographer"]) == k]
        st = [r for r in sup if T.fold_of(r["photographer"]) == k]
        fs = select_b(tr, seed, b)
        row = {"fold": k, "tau_g": fs["tau_g"], "tau_c": fs["tau_c"], "heldout_unsupported": len(te), "heldout_supported": len(st)}
        if fs["tau_g"] is not None:
            row["heldout_fa"] = sum(accept_b(r["top1"], r["score"], fs["tau_g"], fs["tau_c"]) for r in te) / len(te) if te else None
            row.update({f"heldout_{a}": v for a, v in _sup_metrics_b(st, fs["tau_g"], fs["tau_c"]).items() if a != "n"})
        folds.append(row)
    problems = []
    if supm["coverage"] < 0.70:
        problems.append(f"supported coverage {supm['coverage']:.3f} < 0.70")
    if supm["correct_with_refusals"] < 0.80:
        problems.append(f"supported correct-with-refusals {supm['correct_with_refusals']:.3f} < 0.80")
    if any(f["tau_g"] is None for f in folds):
        problems.append("a fold could not select a threshold")
    else:
        for key in ("tau_g", "tau_c"):
            v = [f[key] for f in folds]
            if max(v) - min(v) > 0.05:
                problems.append(f"unstable: fold {key} range {max(v) - min(v):.3f} > 0.05")
    over = sum(1 for f in folds if f.get("heldout_fa") is not None and f["heldout_fa"] > T.TARGET)
    if over > 1:
        problems.append(f"unstable: out-of-fold false acceptance > 10% in {over} folds")
    acc_fn = lambda r: accept_b(r["top1"], r["score"], tg, tc)  # noqa: E731
    return {"candidate": "B", "status": "resolved" if not problems else "unresolved", "reason": "; ".join(problems) or None,
            "tau_g": tg, "tau_c": tc, "selection": sel, "supported": supm, "folds": folds, "minimums": mins,
            "supported_by_taxon": by_taxon(sup, acc_fn), "unsupported_by_group": by_group(uns, acc_fn)}


# ------------------------------------------------------------------------------------------------------ breakdowns
def by_taxon(sup: list[dict], accepted) -> dict:
    out = {}
    for r in sup:
        d = out.setdefault(r["taxon"], {"n": 0, "answered": 0, "correct": 0})
        d["n"] += 1
        if accepted(r):
            d["answered"] += 1
            d["correct"] += int(r["correct"])
    return out


def by_group(uns: list[dict], accepted) -> dict:
    out = {}
    for r in uns:
        d = out.setdefault(r["group"], {"n": 0, "accepted": 0, "accepted_as": {}})
        d["n"] += 1
        if accepted(r):
            d["accepted"] += 1
            d["accepted_as"][r["top1"]] = d["accepted_as"].get(r["top1"], 0) + 1
    return out


def evaluate_all(recs: list[dict], names: list[str], seed: int = T.SEED, b: int = T.B) -> dict:
    """Every pre-declared candidate, then the pre-declared adoption rule (first passing in order A, B, C)."""
    res = {"A": run_single(recs, names, "A"), "B": run_b(recs, names, seed, b), "C": run_single(recs, names, "C")}
    adopted = next((c for c in CANDIDATES if res[c]["status"] == "resolved"), None)
    return {"candidates": res, "adopted": adopted, "status": "resolved" if adopted else "unresolved"}


# ------------------------------------------------------------------------------------------- runtime decision rule
def decide(probs: list[float], names: list[str], op: dict | None) -> dict:
    """Apply an adopted v3 operating point to one image. Never answers without a resolved operating point."""
    if not op or op.get("status") != "resolved" or op.get("adopted") not in CANDIDATES:
        return {"outcome": "not_run", "taxon": None}
    if probs is None or len(probs) != len(names):
        return {"outcome": "not_run", "taxon": None}
    rec = {"probs": probs}
    t, s = top1(rec, names)
    c = op["adopted"]
    if c == "A":
        ok = score_a(rec, names) >= op["tau"]
    elif c == "C":
        ok = score_c(rec, names) >= op["tau"]
    else:
        ok = accept_b(t, s, op["tau_g"], op["tau_c"])
    return {"outcome": "suggestion" if ok else "unidentified", "taxon": t if ok else None}
