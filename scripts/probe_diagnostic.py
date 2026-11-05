"""Part B (Stage 7) — Diagnostic representation probe — not an OSR method.

    python scripts/probe_diagnostic.py                    # original Stage-3 representation (Stage-6 embeddings)
    python scripts/probe_diagnostic.py --rep contrastive  # after Stage-7 training (results/embeddings_contrastive)

Leaf-grouped 5-fold CV logistic regression on frozen embeddings, for the pairs in
config.yaml -> probe_diagnostic, plus a known-vs-known control and a random-halves control.
Output: results/metrics/probe_diagnostic_<rep>.json, results/tables/probe_diagnostic_<rep>.csv|md
Nothing here feeds any threshold or OSR metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.osr.probe import LABEL, probe  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.tables import write_table  # noqa: E402


def load_rep(cfg, rep: str):
    """All splits of one representation as (meta, E) with a leakage-safe group id per image."""
    d = resolve(cfg["feature_osr"]["embeddings_dir"] if rep == "original" else "results/embeddings_contrastive")
    man = pd.read_csv(resolve("data/splits/manifest_all.csv"), low_memory=False, keep_default_na=False)
    man = man.set_index("image_path")
    metas, Es = [], []
    for s in ("train", "val", "test"):
        z = np.load(d / f"{s}_embeddings.npz", allow_pickle=False)
        meta = pd.read_csv(d / f"{s}_meta.csv", low_memory=False, keep_default_na=False)
        assert list(z["image_path"]) == meta["image_path"].tolist()
        sg = man.loc[meta["image_path"], "split_group_id"].to_numpy()
        lg = man.loc[meta["image_path"], "leaf_group_id"].to_numpy()
        # known PV rows: merged leaf+duplicate group; other PV rows: leaf group; always class-prefixed
        meta["group"] = [f"{c}||{a or b or p}" for c, a, b, p in zip(meta["class_name"], sg, lg, meta["image_path"])]
        metas.append(meta); Es.append(z["embeddings"].astype(np.float64))
    meta = pd.concat(metas, ignore_index=True)
    E = np.concatenate(Es)
    if rep != "original":
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return meta, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="original", choices=["original", "contrastive"])
    a = ap.parse_args()
    cfg = load_config()
    pc = cfg["probe_diagnostic"]
    meta, E = load_rep(cfg, a.rep)
    pv = (meta["dataset"] == "plantvillage").to_numpy()
    sel = lambda c: np.flatnonzero(pv & (meta["class_name"] == c).to_numpy())  # noqa: E731
    out, rows = {"label": LABEL, "representation": a.rep, "results": {}}, []

    def run(name, ia, ib):
        r = probe(E[ia], E[ib], meta["group"].to_numpy()[ia], meta["group"].to_numpy()[ib],
                  C=pc["C"], repeats=pc["repeats"], n_boot=pc["bootstrap"], seed=cfg["seed"])
        out["results"][name] = r
        rows.append({"comparison": name, "n_pos": r["n_pos"], "n_neg": r["n_neg"], "groups": r["n_groups"],
                     "auroc_mean": r["auroc_mean"], "auroc_ci95_lo": r["auroc_group_bootstrap_ci95"][0],
                     "auroc_ci95_hi": r["auroc_group_bootstrap_ci95"][1], "accuracy_mean": r["accuracy_mean"],
                     "f1_mean": r["f1_mean"], "auroc_ungrouped_cv": r["auroc_ungrouped_cv_for_reference"]})
        print(f"  {name:70s} AUROC {r['auroc_mean']:.4f} {r['auroc_group_bootstrap_ci95']} acc {r['accuracy_mean']:.4f} "
              f"F1 {r['f1_mean']:.4f} | ungrouped CV {r['auroc_ungrouped_cv_for_reference']:.4f}")

    for u, k in pc["pairs"]:
        run(f"{u} vs {k}", sel(u), sel(k))
    a_, b_ = pc["controls"]["known_pair"]
    run(f"CONTROL known-vs-known: {a_} vs {b_}", sel(a_), sel(b_))
    ix = sel(pc["controls"]["random_half_class"])
    groups = meta["group"].to_numpy()[ix]
    ug = np.unique(groups)
    half = set(np.random.default_rng(cfg["seed"]).permutation(ug)[: len(ug) // 2])
    m = np.array([g in half for g in groups])
    run(f"CONTROL random halves (by leaf group) of {pc['controls']['random_half_class']}", ix[m], ix[~m])

    (resolve(cfg["paths"]["metrics"]) / f"probe_diagnostic_{a.rep}.json").write_text(json.dumps(out, indent=2))
    write_table(pd.DataFrame(rows).round(4), resolve(cfg["paths"]["tables"]) / f"probe_diagnostic_{a.rep}")
    print(LABEL)


if __name__ == "__main__":
    main()
