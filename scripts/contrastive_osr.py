"""Stage 7 — evaluate the contrastive representation against the original one, same protocol.

    python scripts/contrastive_osr.py extract    # 2048-d backbone features (projection head removed), all splits
    python scripts/contrastive_osr.py select     # TRAIN-only fits + KNOWN-validation thresholds (frozen file)
    python scripts/contrastive_osr.py evaluate   # protocol check, test scored once, analyses, figures
    python scripts/contrastive_osr.py report     # results/reports/STAGE_7_REPRESENTATION_LEARNING.md

Pre-registered grid: representation {original (Stage 3), contrastive (Stage 7)} x score
{prototype_cosine = 1 - max cos to train prototypes, mahalanobis_normalised (tied Ledoit-Wolf on
L2-normalised features)}; plus MSP/Energy of a linear head fitted on frozen contrastive TRAIN features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.osr_msp import eval_pair, manifest, sha256  # noqa: E402
from src.data.dataset import ManifestDataset, build_transforms  # noqa: E402
from src.data.protocol import known_classes, verify_lock  # noqa: E402
from src.evaluation.bootstrap import paired_bootstrap  # noqa: E402
from src.evaluation.osr_metrics import ranking_metrics  # noqa: E402
from src.osr.mahalanobis import TiedMahalanobis  # noqa: E402
from src.osr.scores import energy_score, msp_scores  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.tables import write_table  # noqa: E402

CDIR = "results/embeddings_contrastive"
THR_FILE = "contrastive_thresholds.json"
TEST_LOG = "results/test_runs.log"
META = ["image_path", "dataset", "crop", "disease", "class_name", "class_id", "role", "known_unknown", "split",
        "pv_equivalent", "strict_test", "eval_subset", "experiments"]


def l2n(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ----------------------------------------------------------------------------- extract
def cmd_extract(cfg):
    from scripts.train_contrastive import ContrastiveNet
    verify_lock(cfg)
    c, b = cfg["contrastive"], cfg["baseline"]
    ck = torch.load(resolve(cfg["paths"]["models"]) / c["run_name"] / "best_model.pt", map_location="cpu", weights_only=False)
    model = ContrastiveNet(21, c["trainable"], c["projection"])
    model.load_state_dict(ck["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tf = build_transforms(b["training"], b["augmentation"], train=False)       # same eval preprocessing as Stage 3
    out = resolve(CDIR); out.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run(ds):
        F_ = []
        for x, _, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=6, pin_memory=True):
            F_.append(model.features(x.to(device)).float().cpu())              # fp32, projection head NOT used
        return torch.cat(F_).numpy()

    checks = {"representation": "backbone 2048-d global-avg-pooled layer4 (projection head removed)", "splits": {}}
    for split in ("train", "val", "test"):
        df = manifest(f"data/splits/{split}.csv", split)
        E = run(ManifestDataset(df, tf))
        assert E.shape == (len(df), 2048) and np.isfinite(E).all()
        np.savez_compressed(out / f"{split}_embeddings.npz", embeddings=E.astype(np.float32),
                            image_path=np.array(df["image_path"].tolist()))
        df[[m for m in META if m in df.columns]].to_csv(out / f"{split}_meta.csv", index=False)
        checks["splits"][split] = {"n": len(df), "finite": True, "norm_median": float(np.median(np.linalg.norm(E, axis=1)))}
        print(f"  {split}: {E.shape}")
    df = manifest("data/splits/test.csv", "test")
    E2 = run(Subset(ManifestDataset(df, tf), list(range(64))))
    E1 = np.load(out / "test_embeddings.npz")["embeddings"][:64]
    checks["determinism_rerun_max_abs_diff"] = float(np.abs(E1 - E2).max())
    checks["checkpoint_sha256"] = sha256(resolve(cfg["paths"]["models"]) / c["run_name"] / "best_model.pt")
    checks["best_epoch"] = ck["best_epoch"]
    (out / "extraction_checks.json").write_text(json.dumps(checks, indent=2))
    print(checks)


def load_rep(cfg, rep, split):
    d = resolve(cfg["feature_osr"]["embeddings_dir"] if rep == "original" else CDIR)
    z = np.load(d / f"{split}_embeddings.npz", allow_pickle=False)
    meta = pd.read_csv(d / f"{split}_meta.csv", low_memory=False, keep_default_na=False)
    assert list(z["image_path"]) == meta["image_path"].tolist()
    meta["class_id"] = meta["class_id"].astype(int)
    meta["strict_test"] = meta["strict_test"].astype(str).str.lower().eq("true")
    return meta, z["embeddings"].astype(np.float64)


class Fits:
    """All TRAIN-only fits for one representation."""

    def __init__(self, cfg, rep, linear_head: bool):
        tr, E = load_rep(cfg, rep, "train")
        assert set(tr["role"]) == {"known"}
        X, y = l2n(E), tr["class_id"].to_numpy()
        P = np.stack([X[y == k].mean(0) for k in range(21)])
        self.protos = l2n(P)
        self.maha = TiedMahalanobis(cfg["feature_osr"]["eig_floor_rel"]).fit(X, y, 21)
        self.lin = None
        if linear_head:
            lh = cfg["contrastive"]["linear_head"]
            self.lin = LogisticRegression(C=lh["C"], max_iter=lh["max_iter"]).fit(X, y)
        self.train_n = len(tr)

    def scores(self, E):
        X = l2n(E)
        sim = X @ self.protos.T
        out = {"prototype_cosine": 1 - sim.max(1), "proto_class": sim.argmax(1), "cos_all": sim}
        out["mahalanobis_normalised"], out["maha_class"] = self.maha.score(X)
        if self.lin is not None:
            L = self.lin.decision_function(X)
            ms = msp_scores(L)
            out.update({"msp_linear_head_u": ms["u"], "msp_linear_head": ms["msp"], "energy_linear_head": energy_score(L, 1.0),
                        "linear_pred": ms["pred"]})
        return out


SCORE_KEYS = {"original": ["prototype_cosine", "mahalanobis_normalised"],
              "contrastive": ["prototype_cosine", "mahalanobis_normalised", "msp_linear_head_u", "energy_linear_head"]}


# ----------------------------------------------------------------------------- select (validation only)
def cmd_select(cfg):
    verify_lock(cfg)
    out = resolve(cfg["paths"]["metrics"]) / THR_FILE
    if out.exists():
        raise SystemExit(f"{out} exists — Stage-7 thresholds are frozen.")
    q = cfg["contrastive"]["target_known_acceptance"]
    rec = {"rule": "accept as known iff score <= threshold; threshold accepts 95% of KNOWN validation images",
           "fits": "prototypes / Mahalanobis / linear head fitted on KNOWN TRAINING embeddings only", "thresholds": {},
           "known_validation": {}, "validation_unknown_reported_only": {}}
    for rep in ("original", "contrastive"):
        f = Fits(cfg, rep, linear_head=rep == "contrastive")
        va, Ev = load_rep(cfg, rep, "val")
        s = f.scores(Ev)
        k = (va["role"] == "known").to_numpy(); uv = (va["role"] == "unknown_val").to_numpy()
        y = va["class_id"].to_numpy()[k]
        kv = {"nc_accuracy": float(accuracy_score(y, s["proto_class"][k])),
              "nc_macro_f1": float(f1_score(y, s["proto_class"][k], average="macro"))}
        if f.lin is not None:
            kv.update({"linear_head_accuracy": float(accuracy_score(y, s["linear_pred"][k])),
                       "linear_head_macro_f1": float(f1_score(y, s["linear_pred"][k], average="macro"))})
        rec["known_validation"][rep] = kv
        for key in SCORE_KEYS[rep]:
            t = float(np.quantile(s[key][k], q, method="higher"))
            rec["thresholds"][f"{rep}::{key}"] = t
            rec["validation_unknown_reported_only"][f"{rep}::{key}"] = {
                "auroc_known_val_vs_validation_unknown": ranking_metrics(s[key][k], s[key][uv])["auroc"],
                "validation_unknown_rejected": float((s[key][uv] > t).mean())}
    rec.update({"checkpoint_sha256": sha256(resolve(cfg["paths"]["models"]) / cfg["contrastive"]["run_name"] / "best_model.pt"),
                "contrastive_test_embeddings_sha256": sha256(resolve(CDIR) / "test_embeddings.npz"),
                "test_used": False, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec["known_validation"], indent=2))
    for k, v in rec["validation_unknown_reported_only"].items():
        print(f"  {k:42s} tau {rec['thresholds'][k]:.6g}  val-unknown AUROC {v['auroc_known_val_vs_validation_unknown']:.4f}")


# ----------------------------------------------------------------------------- evaluate
def protocol_check(cfg, thr):
    tr, te = manifest("data/splits/train.csv", "train"), manifest("data/splits/test.csv", "test")
    cm = yaml.safe_load(resolve("data/splits/class_mapping.yaml").read_text(encoding="utf-8"))
    near = [c["class_name"] for c in cm["unknown_classes"]["unknown_near (Experiment B, test only)"]]
    log = resolve(TEST_LOG)
    prior = [l for l in log.read_text().splitlines() if "stage7" in l] if log.exists() else []
    hist = json.loads((resolve(cfg["paths"]["metrics"]) / f"{cfg['contrastive']['run_name']}_training_history.json").read_text())
    checks = {
        "Known classes in contrastive training": tr["class_name"].nunique(),
        "Unknown classes in contrastive training": len(set(tr["class_name"]) & (set(cfg["osr_protocol"]["unknown_near"]) | set(cfg["osr_protocol"]["unknown_val"]))),
        "Hard-unknown classes equal locked class_mapping": "YES" if near == list(cfg["osr_protocol"]["unknown_near"]) else "NO",
        "Checkpoint selected on known validation only": "YES" if all("val_nearest_centroid_macro_f1" in h for h in hist["history"]) else "NO",
        "Thresholds selected from known validation": "YES" if not thr["test_used"] else "NO",
        "Final test used for selection": "NO" if all(l.split()[0] > thr["created_utc"] for l in prior) else "YES",
        "Contrastive checkpoint unchanged since selection": "YES" if sha256(resolve(cfg["paths"]["models"]) / cfg["contrastive"]["run_name"] / "best_model.pt") == thr["checkpoint_sha256"] else "NO",
        "Stage-3 checkpoint unchanged": "YES" if sha256(resolve(cfg["osr_msp"]["checkpoint"])) == cfg["osr_msp"]["checkpoint_sha256"] else "NO",
    }
    exp = {"Known classes in contrastive training": 21, "Unknown classes in contrastive training": 0,
           "Hard-unknown classes equal locked class_mapping": "YES", "Checkpoint selected on known validation only": "YES",
           "Thresholds selected from known validation": "YES", "Final test used for selection": "NO",
           "Contrastive checkpoint unchanged since selection": "YES", "Stage-3 checkpoint unchanged": "YES"}
    print("\nPROTOCOL CHECK")
    for k, v in checks.items():
        print(f"  {k}: {v}" + ("" if v == exp[k] else "   <-- FAIL"))
    if checks != exp:
        raise SystemExit("STOP: protocol check failed")
    return checks


def rep_diagnostics(E, meta, protos):
    """Known-test geometry on L2-normalised features (centroids = TRAIN prototypes)."""
    k = (meta["role"] == "known").to_numpy()
    X, y = l2n(E[k]), meta["class_id"].to_numpy()[k]
    own = (X * protos[y]).sum(1)
    cc = protos @ protos.T
    off = cc[~np.eye(21, dtype=bool)]
    intra = 1 - own                                   # cosine distance to own prototype
    inter = 1 - off                                   # cosine distance between prototypes
    return {"intra_class_cosine_distance_mean": float(intra.mean()), "intra_class_cosine_distance_median": float(np.median(intra)),
            "inter_prototype_cosine_distance_mean": float(inter.mean()), "inter_prototype_cosine_distance_min": float(inter.min()),
            "separation_ratio_inter_mean_over_intra_mean": float(inter.mean() / intra.mean()),
            "nearest_prototype_accuracy_known_test": float(((X @ protos.T).argmax(1) == y).mean()),
            "nearest_other_prototype_margin_mean": float((own - np.where(np.eye(21, dtype=bool)[y], -9, X @ protos.T).max(1)).mean())}


def cmd_evaluate(cfg):
    thr = json.loads((resolve(cfg["paths"]["metrics"]) / THR_FILE).read_text())
    checks = protocol_check(cfg, thr)
    classes = known_classes(cfg)
    near = cfg["osr_protocol"]["unknown_near"]
    with open(resolve(TEST_LOG), "a") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} stage7 contrastive evaluate "
                 f"thresholds={json.dumps(thr['thresholds'])} test_sha256={sha256(resolve('data/splits/test.csv'))}\n")
    co = cfg["contrastive"]
    rng = np.random.default_rng(co["seed"])
    res, dfs, diag, fits = {}, {}, {}, {}
    for rep in ("original", "contrastive"):
        f = Fits(cfg, rep, linear_head=rep == "contrastive")
        fits[rep] = f
        te, E = load_rep(cfg, rep, "test")
        s = f.scores(E)
        df = te.assign(label=te["class_id"].to_numpy(), cos_all=list(s["cos_all"]))
        for key in SCORE_KEYS[rep]:
            df[key] = s[key]
        if "msp_linear_head" in s:
            df["msp_linear_head"] = s["msp_linear_head"]
        df["proto_class"] = [classes[i] for i in s["proto_class"]]
        if rep == "contrastive":
            df["pred"] = s["linear_pred"]
        else:
            z = np.load(resolve(cfg["osr_energy"]["logits_test"]))
            assert list(z["paths"]) == te["image_path"].tolist()
            ms = msp_scores(z["logits"])
            df["pred"], df["u_msp"], df["energy"] = ms["pred"], ms["u"], energy_score(z["logits"], 1.0)
        df["pred_class"] = [classes[i] for i in df["pred"]]
        dfs[rep] = df
        diag[rep] = rep_diagnostics(E, te, f.protos)
        for key in SCORE_KEYS[rep]:
            d = df.assign(u=df[key])
            K = d[d["role"] == "known"]
            sets = {"B_hard_same_crop": (K, d[d["role"] == "unknown_near"]),
                    "A1_unseen_crops": (K, d[d["role"] == "unknown_far_crop"]),
                    "A2_non_plant_textures": (K, d[d["role"] == "unknown_far_texture"]),
                    "B_strict_known_sensitivity": (K[K["strict_test"]], d[d["role"] == "unknown_near"]),
                    "C_plantdoc_all_unknown": (d[d["role"] == "known_field"], d[d["role"].isin(["unknown_near_field", "unknown_far_field"])]),
                    "C_plantdoc_near_unknown": (d[d["role"] == "known_field"], d[d["role"] == "unknown_near_field"]),
                    "C_plantdoc_far_unknown": (d[d["role"] == "known_field"], d[d["role"] == "unknown_far_field"])}
            res[f"{rep}::{key}"] = {k: eval_pair(kd, ud, thr["thresholds"][f"{rep}::{key}"], 21) for k, (kd, ud) in sets.items()}
        print(f"  scored {rep}")
    o, c = dfs["original"], dfs["contrastive"]
    assert (o["image_path"].to_numpy() == c["image_path"].to_numpy()).all()
    s5 = json.loads(resolve(cfg["osr_energy"]["stage4_results"]).read_text())
    s5e = json.loads(resolve(cfg["feature_osr"]["stage5_results"]).read_text())
    s6 = json.loads((resolve(cfg["paths"]["metrics"]) / "mahalanobis_results.json").read_text())
    prior = {"original::msp": s5["results"], "original::energy": s5e["energy_results"]["primary"],
             "original::mahalanobis_raw": s6["mahalanobis_results"]}
    t_en = json.loads(resolve(cfg["feature_osr"]["stage5_threshold"]).read_text())["primary"]["threshold_energy"]
    T = thr["thresholds"]

    # paired bootstraps on identical images
    def bp(a_df, a_key, a_t, b_df, b_key, b_t, unk_mask_fn):
        ka, kb = a_df[a_df["role"] == "known"], b_df[b_df["role"] == "known"]
        ua, ub = a_df[unk_mask_fn(a_df)], b_df[unk_mask_fn(b_df)]
        return paired_bootstrap(ka[a_key].to_numpy(), ua[a_key].to_numpy(), a_t, kb[b_key].to_numpy(), ub[b_key].to_numpy(), b_t,
                                co["bootstrap_resamples"], co["seed"])
    B = lambda d: d["role"] == "unknown_near"  # noqa: E731
    boot = {
        "proto: original vs contrastive (B)": bp(o, "prototype_cosine", T["original::prototype_cosine"], c, "prototype_cosine", T["contrastive::prototype_cosine"], B),
        "maha_norm: original vs contrastive (B)": bp(o, "mahalanobis_normalised", T["original::mahalanobis_normalised"], c, "mahalanobis_normalised", T["contrastive::mahalanobis_normalised"], B),
        "original Energy vs contrastive proto (B)": bp(o, "energy", t_en, c, "prototype_cosine", T["contrastive::prototype_cosine"], B),
        "original Energy vs contrastive Energy-head (B)": bp(o, "energy", t_en, c, "energy_linear_head", T["contrastive::energy_linear_head"], B),
        "proto: original vs contrastive (A1)": bp(o, "prototype_cosine", T["original::prototype_cosine"], c, "prototype_cosine", T["contrastive::prototype_cosine"], lambda d: d["role"] == "unknown_far_crop"),
    }
    for cl in near:
        boot[f"proto: original vs contrastive (B class {cl})"] = bp(o, "prototype_cosine", T["original::prototype_cosine"], c, "prototype_cosine",
                                                                     T["contrastive::prototype_cosine"], lambda d, cl=cl: d["class_name"] == cl)
    print("  bootstraps done")

    # per-disease table
    rows = []
    for cl in near:
        r = {"unknown_class": cl, "images": int((o["class_name"] == cl).sum())}
        for name, R in {**{k: v for k, v in res.items()}, **prior}.items():
            pc = {x["class_name"]: x for x in R["B_hard_same_crop"]["per_unknown_class"]}[cl]
            r[f"{name}::auroc"] = pc["auroc_vs_known"]; r[f"{name}::detection"] = pc["unknown_recall"]
        rows.append(r)
    per_cls = pd.DataFrame(rows)

    # Esca / NLB deep comparison
    def deep(unknown, look):
        out = {}
        li = classes.index(look)
        for rep, d in dfs.items():
            u, kk = d[d["class_name"] == unknown], d[d["role"] == "known"]
            lk = kk[kk["label"] == li]
            cu, cl_ = np.stack(u["cos_all"]), np.stack(lk["cos_all"])
            key_t = {k: T[f"{rep}::{k}"] for k in ("prototype_cosine", "mahalanobis_normalised")}
            out[rep] = {"share_nearest_prototype_is_lookalike": float((cu.argmax(1) == li).mean()),
                        "mean_cosine_unknown_to_lookalike_prototype": float(cu[:, li].mean()),
                        "mean_cosine_lookalike_test_to_own_prototype": float(cl_[:, li].mean()),
                        "mean_nearest_prototype_cosine_distance_unknown": float(u["prototype_cosine"].mean()),
                        "mean_nearest_prototype_cosine_distance_known_test": float(kk["prototype_cosine"].mean()),
                        **{f"{k}::auroc_vs_all_known": ranking_metrics(kk[k].to_numpy(), u[k].to_numpy())["auroc"] for k in key_t},
                        **{f"{k}::auroc_vs_lookalike_only": ranking_metrics(lk[k].to_numpy(), u[k].to_numpy())["auroc"] for k in key_t},
                        **{f"{k}::detection": float((u[k] > key_t[k]).mean()) for k in key_t},
                        **{f"{k}::false_known_rate": float((u[k] <= key_t[k]).mean()) for k in key_t}}
        return out
    deep_res = {"Grape___Esca_(Black_Measles)": deep("Grape___Esca_(Black_Measles)", "Grape___Black_rot"),
                "Corn_(maize)___Northern_Leaf_Blight": deep("Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")}

    # failure analysis: original best (Energy) vs contrastive primary (prototype)
    U = c[c["role"] == "unknown_near"].index
    od = np.where(o.loc[U, "energy"] <= t_en, "accept", "reject")
    cd = np.where(c.loc[U, "prototype_cosine"] <= T["contrastive::prototype_cosine"], "accept", "reject")
    cat = np.select([(od == "accept") & (cd == "reject"), (od == "reject") & (cd == "accept"),
                     (od == "accept") & (cd == "accept"), (od == "reject") & (cd == "reject")],
                    ["original_fails_contrastive_succeeds", "original_succeeds_contrastive_fails", "both_fail", "both_succeed"])
    fa = pd.DataFrame({"image_path": c.loc[U, "image_path"], "true_unknown": c.loc[U, "class_name"],
                       "original_pred": o.loc[U, "pred_class"], "original_energy": o.loc[U, "energy"],
                       "contrastive_nearest_prototype": c.loc[U, "proto_class"], "contrastive_proto_score": c.loc[U, "prototype_cosine"],
                       "original_energy_decision": od, "contrastive_proto_decision": cd, "category": cat})
    tdir = resolve(cfg["paths"]["tables"])
    fa.to_csv(tdir / "contrastive_vs_original_failures.csv", index=False)
    counts = fa.groupby(["true_unknown", "category"]).size().unstack(fill_value=0)
    ex = fa.groupby(["true_unknown", "category"], group_keys=False).apply(
        lambda g: g.sample(min(2, len(g)), random_state=co["seed"])).reset_index(drop=True)
    write_table(ex, tdir / "contrastive_failure_examples")
    write_table(per_cls.round(6), tdir / "contrastive_per_unknown_class")

    # predictions file
    tc = T["contrastive::prototype_cosine"]
    is_known = c["known_unknown"] == "known"
    pku = np.where(c["prototype_cosine"] <= tc, "known", "unknown")
    lab = np.where(c["dataset"] == "plantdoc", c["pv_equivalent"], c["class_name"])
    pd.DataFrame({"image_path": c["image_path"], "dataset": c["dataset"], "role": c["role"],
                  "true_class": np.where(is_known, lab, c["class_name"]), "true_known_unknown": c["known_unknown"],
                  "predicted_class": np.where(pku == "known", c["proto_class"], "UNKNOWN"),
                  "contrastive_prototype_score": c["prototype_cosine"], "contrastive_mahalanobis_normalised": c["mahalanobis_normalised"],
                  "contrastive_linear_head_msp": c["msp_linear_head"],
                  "contrastive_linear_head_energy": c["energy_linear_head"], "prototype_threshold": tc,
                  "predicted_known_unknown": pku, "correct_osr_decision": pku == c["known_unknown"],
                  "original_energy": o["energy"], "strict_test": c["strict_test"]}).to_csv(
        resolve(cfg["paths"]["results"]) / "predictions" / "osr_contrastive_predictions.csv", index=False)

    # closed-set known accuracy on test
    kc = c[c["role"] == "known"]
    known_test = {"contrastive_linear_head_accuracy": float((kc["pred"] == kc["label"]).mean()),
                  "contrastive_linear_head_macro_f1": float(f1_score(kc["label"], kc["pred"], average="macro")),
                  "contrastive_nearest_prototype_accuracy": diag["contrastive"]["nearest_prototype_accuracy_known_test"],
                  "original_stage3_head_accuracy": s5["closed_set_system"]["known_accuracy_pv"],
                  "original_nearest_prototype_accuracy": diag["original"]["nearest_prototype_accuracy_known_test"],
                  "plantdoc_known_field_contrastive_linear_head_accuracy": float(
                      (c.loc[c["role"] == "known_field", "pred"] == c.loc[c["role"] == "known_field", "label"]).mean()),
                  "plantdoc_known_field_original_accuracy": s5["closed_set_system"]["known_accuracy_plantdoc_field"]}

    out = {"protocol_check": checks, "thresholds": thr, "results": res, "prior_original_results": {k: {e: {m: v[e][m] for m in
           ("auroc", "aupr_out", "fpr_at_95tpr", "unknown_recall", "unknown_precision", "unknown_f1", "known_acceptance_rate",
            "false_rejection_rate", "oscr")} for e in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures",
           "C_plantdoc_all_unknown", "C_plantdoc_near_unknown", "C_plantdoc_far_unknown")} for k, v in prior.items()},
           "bootstrap": boot, "per_unknown_class": per_cls.to_dict(orient="records"), "representation_diagnostics": diag,
           "deep": deep_res, "failure_counts": counts.to_dict(orient="index"), "known_test": known_test,
           "training": json.loads((resolve(cfg["paths"]["metrics"]) / f"{co['run_name']}_training_history.json").read_text()),
           "environment": json.loads((resolve(cfg["paths"]["metrics"]) / f"{co['run_name']}_environment.json").read_text()),
           "extraction_checks": json.loads((resolve(CDIR) / "extraction_checks.json").read_text()),
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (resolve(cfg["paths"]["metrics"]) / "contrastive_results.json").write_text(json.dumps(out, indent=2, default=float))

    from src.visualization import contrastive_plots as CP
    figs = resolve(cfg["paths"]["figures"]) / "contrastive"
    CP.comparison(res, prior, figs / "osr_comparison.png")
    CP.per_disease(per_cls, figs / "per_disease_auroc.png")
    CP.geometry(diag, figs / "representation_geometry.png")
    CP.deep_dive(dfs, near, T, figs / "esca_nlb_original_vs_contrastive.png")
    for name in ("original::prototype_cosine", "original::mahalanobis_normalised", "contrastive::prototype_cosine",
                 "contrastive::mahalanobis_normalised", "contrastive::msp_linear_head_u", "contrastive::energy_linear_head"):
        r = res[name]["B_hard_same_crop"]
        print(f"{name:38s} B AUROC {r['auroc']:.4f} det {r['unknown_recall']:.4f} F1 {r['unknown_f1']:.4f} "
              f"FPR95 {r['fpr_at_95tpr']:.4f} acc {r['known_acceptance_rate']:.4f} | A1 {res[name]['A1_unseen_crops']['auroc']:.4f} "
              f"C {res[name]['C_plantdoc_all_unknown']['auroc']:.4f}")
    print(per_cls.filter(regex="unknown_class|::auroc").round(4).to_string(index=False))
    print(json.dumps(diag, indent=1)); print(known_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "select", "evaluate", "report"])
    a = ap.parse_args()
    cfg = load_config()
    if a.step == "report":
        from scripts.contrastive_report import write_report
        write_report(cfg)
    else:
        {"extract": cmd_extract, "select": cmd_select, "evaluate": cmd_evaluate}[a.step](cfg)

if __name__ == "__main__":
    main()
