"""Tune the decision rule on validation, then evaluate it on the test set (once).

Fitted on validation only (stage11/decision.py explains the rule):
  T     temperature, minimising the NLL of the crop-restricted softmax;
  beta  symptom-match exponent, minimising the NLL of the fused probability (grid 0, .25, .5, 1, 2);
  m0    photo-fit level = 5th percentile of the in-crop probability mass on field validation photos, so 95% of
        genuine field photos of the stated crop are not penalised.
Test metrics per source (PlantVillage lab, PlantDoc field, apple field = AppleLeaf9 + Kashmir), crop given:
  top-1 / top-3, ECE, the 80% rule (coverage = share of photos given a named disease, accuracy of those),
  and for "Unknown disease" photos whether the true disease is in the top three.

Run:  python -P stage11/evaluate.py
Outputs: models/stage11/calibration.json, stage11/results/evaluation.json, stage11/results/EVALUATION.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.decision import THRESHOLD, fuse, softmax  # noqa: E402
from stage11.kb import load_kb  # noqa: E402
from stage11.model import EVAL_TF, LeafNet  # noqa: E402
from stage11.train import MODEL_DIR, RESULTS, ImageSet  # noqa: E402

SOURCES = {"plantvillage": ["plantvillage"], "plantdoc": ["plantdoc"], "apple_field": ["appleleaf9", "kashmir_apple"]}
FIELD = ["plantdoc", "appleleaf9", "kashmir_apple"]


@torch.no_grad()
def collect(model, df, device):
    dl = DataLoader(ImageSet(df, EVAL_TF), batch_size=64, num_workers=2, pin_memory=True)
    L, S = [], []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            lg, sy = model(x.to(device))
        L.append(lg.float().cpu())
        S.append(torch.sigmoid(sy.float()).cpu())
    return torch.cat(L).numpy(), torch.cat(S).numpy()


def crop_mask(kb, labels):
    crops = np.array([d["crop"] for d in kb.model_classes])
    return crops[None, :] == crops[labels][:, None]


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = pred == labels
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > lo) & (conf <= hi)).mean() * abs(acc[(conf > lo) & (conf <= hi)].mean()
                                                                - conf[(conf > lo) & (conf <= hi)].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any()))


def tune(kb, logits, sym, y, field):
    mask = crop_mask(kb, y)
    prof = kb.symptom_targets()
    T = min(np.arange(0.5, 4.01, 0.02), key=lambda t: nll(fuse(logits, sym, mask, prof, T=t)["p_crop"], y))
    beta = min([0.0, 0.25, 0.5, 1.0, 2.0], key=lambda b: nll(fuse(logits, sym, mask, prof, T=T, beta=b)["fused"], y))
    m = fuse(logits[field], sym[field], mask[field], prof, T=T)["in_mass"]
    m0 = float(np.percentile(m, 5))
    return {"temperature": float(T), "beta": float(beta), "m0": round(m0, 4)}


def metrics(kb, logits, sym, y, cal):
    out = fuse(logits, sym, crop_mask(kb, y), kb.symptom_targets(), T=cal["temperature"], beta=cal["beta"],
               m0=cal["m0"])
    conf = out["conf"]
    pred, top = conf.argmax(1), conf.max(1)
    top3 = np.argsort(-conf, 1)[:, :3]
    named = top >= THRESHOLD
    full = softmax(logits / cal["temperature"])
    sym_t = (kb.symptom_targets()[y] >= 0.5).astype(int)
    aucs = [roc_auc_score(sym_t[:, j], sym[:, j]) for j in range(sym_t.shape[1]) if 0 < sym_t[:, j].sum() < len(y)]
    return {
        "n": int(len(y)),
        "top1_all_classes": float((full.argmax(1) == y).mean()),
        "macro_f1_all_classes": float(f1_score(y, full.argmax(1), average="macro")),
        "top1_crop_given": float((pred == y).mean()),
        "top3_crop_given": float((top3 == y[:, None]).any(1).mean()),
        "ece_crop_given": ece(out["fused"], y),
        "named_coverage": float(named.mean()),
        "named_accuracy": float((pred[named] == y[named]).mean()) if named.any() else None,
        "unknown_top3_contains_true": float((top3[~named] == y[~named, None]).any(1).mean()) if (~named).any() else None,
        "photo_fit_penalised": float((out["fit"] < 1).mean()),
        "symptom_macro_auroc_vs_weak_labels": float(np.mean(aucs)) if aucs else None,
    }, pred


def main():
    device = torch.device("cuda")
    kb = load_kb()
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=False)
    model = LeafNet(len(kb.model_classes), len(kb.symptom_ids), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    idx = pd.read_csv(RESULTS / "image_index.csv")

    val = idx[idx.split == "val"].reset_index(drop=True)
    vl, vs = collect(model, val, device)
    vy = val.label.to_numpy()
    cal = tune(kb, vl, vs, vy, val.source.isin(FIELD).to_numpy())
    cal |= {"threshold": THRESHOLD, "fitted_on": "validation only", "checkpoint_epoch": ckpt["epoch"]}
    (MODEL_DIR / "calibration.json").write_text(json.dumps(cal, indent=2))
    print("calibration", cal)

    out = {"calibration": cal, "created": datetime.now().isoformat(timespec="seconds"), "val": {}, "test": {}}
    for name, srcs in SOURCES.items():
        v = val[val.source.isin(srcs)]
        out["val"][name] = metrics(kb, vl[v.index], vs[v.index], vy[v.index], cal)[0]

    test = idx[idx.split == "test"].reset_index(drop=True)
    tl, ts = collect(model, test, device)
    ty = test.label.to_numpy()
    per_class = {}
    for name, srcs in SOURCES.items():
        t = test[test.source.isin(srcs)]
        m, pred = metrics(kb, tl[t.index], ts[t.index], ty[t.index], cal)
        out["test"][name] = m
        if name != "plantvillage":
            for key, g in t.assign(ok=pred == ty[t.index]).groupby("class_key"):
                per_class[f"{name}:{key}"] = {"n": len(g), "acc": round(float(g.ok.mean()), 3)}
    out["test"]["per_class_field"] = per_class
    (RESULTS / "evaluation.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "test_runs.log", "a", encoding="utf-8") as f:
        f.write(f"{out['created']} evaluate.py v2 (43 classes) epoch {ckpt['epoch']} "
                f"T={cal['temperature']:.2f} beta={cal['beta']} m0={cal['m0']}\n")
    write_report(out)
    print(json.dumps({k: v for k, v in out["test"].items() if k != "per_class_field"}, indent=2))


def write_report(out):
    def pct(v):
        return "–" if v is None else f"{100 * v:.1f}%"
    rows = [("Top-1, all classes (crop not given)", "top1_all_classes"),
            ("Top-1, crop given", "top1_crop_given"), ("Top-3, crop given", "top3_crop_given"),
            ("ECE, crop given (lower is better)", "ece_crop_given"),
            ("Photos given a named disease (conf >= 80%)", "named_coverage"),
            ("Accuracy of those named diseases", "named_accuracy"),
            ("'Unknown disease' photos: true disease in top 3", "unknown_top3_contains_true"),
            ("Photos with photo-fit penalty", "photo_fit_penalised"),
            ("Symptom head macro AUROC (vs weak labels)", "symptom_macro_auroc_vs_weak_labels")]
    t = out["test"]
    c = out["calibration"]
    lines = ["# Stage 11 model evaluation (v2, 43 classes)", "",
             f"Generated {out['created']} by `stage11/evaluate.py`. Test set scored once. Fitted on validation only: "
             f"T = {c['temperature']:.2f}, symptom-match beta = {c['beta']}, photo-fit m0 = {c['m0']}. "
             f"Threshold {int(100 * c['threshold'])}%. Weather is not part of this evaluation (no weather labels).", "",
             f"| Metric | PlantVillage lab (n={t['plantvillage']['n']}) | PlantDoc field (n={t['plantdoc']['n']}) "
             f"| Apple field (n={t['apple_field']['n']}) |", "|---|---|---|---|"]
    for name, k in rows:
        lines.append(f"| {name} | {pct(t['plantvillage'][k])} | {pct(t['plantdoc'][k])} | {pct(t['apple_field'][k])} |")
    lines += ["", "## Field test photos, crop given, per class", "", "| Source:class | n | Top-1 |", "|---|---|---|"]
    for k, v in t["per_class_field"].items():
        lines.append(f"| {k} | {v['n']} | {pct(v['acc'])} |")
    lines += ["", "Notes:", "- PlantDoc test is small (~225 photos, 3–12 per class).",
              "- Apple field = AppleLeaf9 (PlantVillage copies removed) + Kashmir indigenous dataset; split by "
              "near-duplicate groups.",
              "- The symptom head is trained on disease-level weak labels; its AUROC against them is a sanity check."]
    (RESULTS / "EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
