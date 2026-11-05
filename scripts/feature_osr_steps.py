"""Stage 6 `select` and `evaluate` steps (see scripts/feature_osr.py)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from scripts.feature_osr import emb_dir, load_split
from scripts.osr_msp import eval_pair, manifest, sha256
from src.data.protocol import known_classes, verify_lock
from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.osr_metrics import ranking_metrics
from src.osr.mahalanobis import TiedMahalanobis
from src.osr.scores import energy_score, msp_scores
from src.utils.config import resolve
from src.utils.tables import write_table

THR_FILE = "mahalanobis_threshold.json"
TEST_LOG = "results/test_runs.log"


def fit_maha(cfg):
    tr, E, _ = load_split(cfg, "train")
    if set(tr["role"]) != {"known"} or set(tr["split"]) != {"train"}:
        raise SystemExit("STOP: Mahalanobis statistics must come from known TRAINING images only")
    m = TiedMahalanobis(cfg["feature_osr"]["eig_floor_rel"]).fit(E, tr["class_id"].to_numpy(), 21)
    return m, tr, E


# ----------------------------------------------------------------------------- select (train fit + validation threshold)
def cmd_select(cfg):
    fo = cfg["feature_osr"]
    verify_lock(cfg)
    out = resolve(cfg["paths"]["metrics"]) / THR_FILE
    if out.exists():
        raise SystemExit(f"{out} exists — the Mahalanobis threshold is frozen.")
    m, tr, Etr = fit_maha(cfg)
    va, Ev, Lv = load_split(cfg, "val")
    d, nearest = m.score(Ev)
    known = (va["role"] == "known").to_numpy()
    uval = (va["role"] == "unknown_val").to_numpy()
    correct = va["pred_class_id"].to_numpy()[known] == va["class_id"].to_numpy()[known]
    tau = float(np.quantile(d[known], fo["target_known_acceptance"], method="higher"))
    s5 = json.loads(resolve(fo["stage5_threshold"]).read_text())
    rec = {
        "threshold_mahalanobis": tau,
        "rule": "accept as known iff min_c squared Mahalanobis distance <= threshold",
        "selection": "threshold = distance accepting 95% of KNOWN validation images (same rule as MSP/Energy); "
                     "class means + tied Ledoit-Wolf covariance from KNOWN TRAINING embeddings only",
        "fit": {**m.summary(), "n_train": int(len(tr)), "train_manifest_sha256": sha256(resolve("data/splits/train.csv")),
                "train_embeddings_sha256": sha256(emb_dir(cfg) / "train_embeddings.npz")},
        "validation": {"n_known": int(known.sum()), "known_val_acceptance": float((d[known] <= tau).mean()),
                       "misclassified_val_rejected": float((d[known][~correct] > tau).mean()),
                       "correct_val_rejected": float((d[known][correct] > tau).mean()),
                       "validation_unknown_rejected_reported_only": float((d[uval] > tau).mean()),
                       "auroc_known_val_vs_validation_unknown_reported_only": ranking_metrics(d[known], d[uval])["auroc"],
                       "maha_nearest_class_agrees_with_argmax": float((nearest[known] == va["pred_class_id"].to_numpy()[known]).mean()),
                       "msp_reference_auroc_kval_vs_valunk": s5["msp_validation_reference"]["auroc_known_val_vs_validation_unknown"],
                       "energy_reference_auroc_kval_vs_valunk": s5["per_temperature_validation"]["1.0"]["auroc_known_val_vs_validation_unknown"]},
        "checkpoint_sha256": cfg["osr_msp"]["checkpoint_sha256"],
        "test_embeddings_sha256": sha256(emb_dir(cfg) / "test_embeddings.npz"),
        "test_used": False, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.write_text(json.dumps(rec, indent=2))
    np.savez_compressed(emb_dir(cfg) / "mahalanobis_fit.npz", means=m.means_, W=m.W_)
    print(json.dumps({k: rec[k] for k in ("threshold_mahalanobis", "fit", "validation")}, indent=2))


# ----------------------------------------------------------------------------- protocol check
def protocol_check(cfg, thr):
    tr, te, va = (manifest(f"data/splits/{s}.csv", s) for s in ("train", "test", "val"))
    cm = yaml.safe_load(resolve("data/splits/class_mapping.yaml").read_text(encoding="utf-8"))
    mapped_near = [c["class_name"] for c in cm["unknown_classes"]["unknown_near (Experiment B, test only)"]]
    log = resolve(TEST_LOG)
    prior = [l for l in log.read_text().splitlines() if "stage6" in l] if log.exists() else []
    unk = set(te.loc[te["known_unknown"] == "unknown", "class_name"]) | set(cfg["osr_protocol"]["unknown_val"])
    checks = {
        "Known classes in training": tr["class_name"].nunique(),
        "Unknown classes in training": len(set(tr["class_name"]) & unk),
        "Hard-unknown classes equal locked class_mapping": "YES" if mapped_near == list(cfg["osr_protocol"]["unknown_near"]) else "NO",
        "Class statistics fitted on known training only": "YES" if thr["fit"]["train_manifest_sha256"] == sha256(resolve("data/splits/train.csv"))
        and thr["fit"]["n_train"] == len(tr) else "NO",
        "Threshold selected from validation": "YES" if not thr["test_used"] else "NO",
        "Final test used for selection": "NO" if all(l.split()[0] > thr["created_utc"] for l in prior) else "YES",
        "Baseline checkpoint unchanged": "YES" if sha256(resolve(cfg["osr_msp"]["checkpoint"])) == thr["checkpoint_sha256"] else "NO",
        "Test embeddings unchanged since selection": "YES" if sha256(emb_dir(cfg) / "test_embeddings.npz") == thr["test_embeddings_sha256"] else "NO",
    }
    expected = {"Known classes in training": 21, "Unknown classes in training": 0,
                "Hard-unknown classes equal locked class_mapping": "YES", "Class statistics fitted on known training only": "YES",
                "Threshold selected from validation": "YES", "Final test used for selection": "NO",
                "Baseline checkpoint unchanged": "YES", "Test embeddings unchanged since selection": "YES"}
    print("\nPROTOCOL CHECK")
    for k, v in checks.items():
        print(f"  {k}: {v}" + ("" if v == expected[k] else "   <-- FAIL"))
    if checks != expected:
        raise SystemExit("STOP: protocol check failed")
    return checks


# ----------------------------------------------------------------------------- analyses
def l2n(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def linear_probe(Xa, Xb, seed):
    """DIAGNOSTIC: 5-fold CV logistic regression AUROC for separating two image sets in feature space."""
    X = np.r_[Xa, Xb]; y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def dist_summary(d):
    d = np.asarray(d)
    return {"mean": float(d.mean()), "median": float(np.median(d)), "q10": float(np.quantile(d, .1)), "q90": float(np.quantile(d, .9))}


def cmd_evaluate(cfg):
    fo = cfg["feature_osr"]
    thr = json.loads((resolve(cfg["paths"]["metrics"]) / THR_FILE).read_text())
    checks = protocol_check(cfg, thr)
    classes = known_classes(cfg)
    with open(resolve(TEST_LOG), "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} stage6 mahalanobis evaluate "
                f"tau={thr['threshold_mahalanobis']!r} test_sha256={sha256(resolve('data/splits/test.csv'))}\n")
    m, tr, Etr = fit_maha(cfg)
    te, Ete, Lte = load_split(cfg, "test")
    va, Eva, _ = load_split(cfg, "val")
    tau = thr["threshold_mahalanobis"]
    rng = np.random.default_rng(fo["seed"])

    # frozen Stage-4/5 scores, from the Stage-4 logits cache (identical images/order)
    z = np.load(resolve(cfg["osr_energy"]["logits_test"]))
    assert list(z["paths"]) == te["image_path"].tolist()
    ms, en = msp_scores(z["logits"]), energy_score(z["logits"], 1.0)
    t_msp = json.loads(resolve(cfg["osr_energy"]["stage4_threshold"]).read_text())["threshold_u"]
    t_en = json.loads(resolve(fo["stage5_threshold"]).read_text())["primary"]["threshold_energy"]
    d_maha, maha_cls = m.score(Ete)
    C = m.means_
    # Euclidean + cosine to class centroids (train means), chunked
    eu = np.empty((len(Ete), 21)); co = np.empty((len(Ete), 21))
    Cn = l2n(C)
    for s in range(0, len(Ete), 4096):
        X = Ete[s:s + 4096]
        eu[s:s + 4096] = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1), 0))
        co[s:s + 4096] = l2n(X) @ Cn.T
    lab_name = np.where(te["dataset"] == "plantdoc", te["pv_equivalent"], te["class_name"])
    df = te.assign(u_msp=ms["u"], msp=ms["msp"], energy=en, maha=d_maha, maha_class=[classes[i] for i in maha_cls],
                   nc_dist=eu.min(1), nc_class=[classes[i] for i in eu.argmin(1)], nc_cos=co.max(1),
                   pred=ms["pred"], pred_class=[classes[i] for i in ms["pred"]], label=te["class_id"].to_numpy(),
                   label_name=lab_name)

    def sets(d):
        K = d[d["role"] == "known"]
        return {"B_hard_same_crop": (K, d[d["role"] == "unknown_near"]),
                "A1_unseen_crops": (K, d[d["role"] == "unknown_far_crop"]),
                "A1_unseen_crops_balanced300": (K, d[(d["role"] == "unknown_far_crop") & (d["eval_subset"] == "far_balanced")]),
                "A2_non_plant_textures": (K, d[d["role"] == "unknown_far_texture"]),
                "B_strict_known_sensitivity": (K[K["strict_test"]], d[d["role"] == "unknown_near"]),
                "C_plantdoc_all_unknown": (d[d["role"] == "known_field"], d[d["role"].isin(["unknown_near_field", "unknown_far_field"])]),
                "C_plantdoc_near_unknown": (d[d["role"] == "known_field"], d[d["role"] == "unknown_near_field"]),
                "C_plantdoc_far_unknown": (d[d["role"] == "known_field"], d[d["role"] == "unknown_far_field"])}

    S = sets(df.assign(u=df["maha"]))
    maha_res = {k: eval_pair(kd, ud, tau, 21) for k, (kd, ud) in S.items()}
    s5 = json.loads(resolve(fo["stage5_results"]).read_text())
    msp_res, en_res = s5["msp_results"], s5["energy_results"]["primary"]
    # sanity: the MSP/Energy recomputed here reproduce Stage 4/5 exactly
    chk = max(abs(eval_pair(*sets(df.assign(u=df["u_msp"]))["B_hard_same_crop"], t_msp, 21)["auroc"] - msp_res["B_hard_same_crop"]["auroc"]),
              abs(eval_pair(*sets(df.assign(u=df["energy"]))["B_hard_same_crop"], t_en, 21)["auroc"] - en_res["B_hard_same_crop"]["auroc"]))
    if chk > 1e-12:
        raise SystemExit("STOP: MSP/Energy not reproduced")

    # paired bootstraps: A = Energy (strongest previous), B = Mahalanobis; and A = MSP, B = Maha for B overall
    boot = {}
    near = cfg["osr_protocol"]["unknown_near"]
    for k in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown",
              "C_plantdoc_near_unknown", "C_plantdoc_far_unknown"):
        kd, ud = S[k]
        boot[f"energy_vs_maha::{k}"] = paired_bootstrap(kd["energy"].to_numpy(), ud["energy"].to_numpy(), t_en,
                                                        kd["maha"].to_numpy(), ud["maha"].to_numpy(), tau,
                                                        fo["bootstrap_resamples"], fo["seed"])
    kd, ud = S["B_hard_same_crop"]
    boot["msp_vs_maha::B_hard_same_crop"] = paired_bootstrap(kd["u_msp"].to_numpy(), ud["u_msp"].to_numpy(), t_msp,
                                                             kd["maha"].to_numpy(), ud["maha"].to_numpy(), tau,
                                                             fo["bootstrap_resamples"], fo["seed"])
    for c in near:
        u1 = ud[ud["class_name"] == c]
        boot[f"energy_vs_maha::B_class::{c}"] = paired_bootstrap(kd["energy"].to_numpy(), u1["energy"].to_numpy(), t_en,
                                                                 kd["maha"].to_numpy(), u1["maha"].to_numpy(), tau,
                                                                 fo["bootstrap_resamples"], fo["seed"])
    print("  bootstraps done")

    # per-disease table (MSP/Energy from Stage 5 saved per-class rows)
    s5pc = pd.DataFrame(s5["per_unknown_class"])
    rows = []
    for exp in ("B_hard_same_crop", "A1_unseen_crops", "C_plantdoc_all_unknown"):
        mm = {r["class_name"]: r for r in maha_res[exp]["per_unknown_class"]}
        for r in s5pc[s5pc["experiment"] == exp].itertuples():
            q = mm[r.unknown_class]
            rows.append({"experiment": exp, "unknown_class": r.unknown_class, "crop": r.crop, "images": r.images,
                         "msp_auroc": r.msp_auroc, "energy_auroc": r.energy_auroc, "mahalanobis_auroc": q["auroc_vs_known"],
                         "msp_detection": r.msp_detection, "energy_detection": r.energy_detection,
                         "mahalanobis_detection": q["unknown_recall"], "mahalanobis_unknown_f1": q["unknown_f1"]})
    per_cls = pd.DataFrame(rows)

    # ---------------- feature-space proximity (train centroids; 1-NN on a seeded train subsample)
    ref_idx = np.concatenate([rng.choice(np.flatnonzero(tr["class_id"].to_numpy() == c),
                                         min(fo["knn_train_reference_per_class"], int((tr["class_id"] == c).sum())), replace=False)
                              for c in range(21)])
    R = l2n(Etr[ref_idx]); ry = tr["class_id"].to_numpy()[ref_idx]
    prox = []
    known_t = df[df["role"] == "known"]
    own = eu[known_t.index.to_numpy(), known_t["label"].to_numpy()]
    for role in ("unknown_near", "unknown_far_crop", "unknown_far_texture"):
        for c, d in df[df["role"] == role].groupby("class_name"):
            ix = d.index.to_numpy()
            nc = pd.Series(eu[ix].argmin(1)).value_counts()
            top = int(nc.index[0])
            cen = Ete[ix].mean(0)
            cen_d = np.sqrt(((C - cen) ** 2).sum(1))
            nn = ry[(l2n(Ete[ix]) @ R.T).argmax(1)]
            row = {"unknown": c, "role": role, "crop": d["crop"].iat[0], "images": len(d),
                   "nearest_known_by_images": classes[top], "share_images_nearest_to_it": float(nc.iat[0] / len(d)),
                   "nearest_known_by_class_centroid": classes[int(cen_d.argmin())],
                   "dist_to_nearest_centroid_mean": float(eu[ix].min(1).mean()),
                   "dist_to_nearest_centroid_median": float(np.median(eu[ix].min(1))),
                   "cosine_to_nearest_centroid_mean": float(co[ix].max(1).mean()),
                   "nn1_train_class_share_of_nearest_known": float((nn == top).mean()),
                   "nn1_train_same_crop_share": float(np.mean([classes[k].split("___")[0] == d["crop"].iat[0] for k in nn]))
                   if role != "unknown_far_texture" else None,
                   "msp_auroc": ranking_metrics(known_t["u_msp"].to_numpy(), d["u_msp"].to_numpy())["auroc"],
                   "energy_auroc": ranking_metrics(known_t["energy"].to_numpy(), d["energy"].to_numpy())["auroc"],
                   "mahalanobis_auroc": ranking_metrics(known_t["maha"].to_numpy(), d["maha"].to_numpy())["auroc"],
                   "nearest_centroid_auroc": ranking_metrics(known_t["nc_dist"].to_numpy(), d["nc_dist"].to_numpy())["auroc"]}
            prox.append(row)
    prox = pd.DataFrame(prox)
    known_ref = {"known_test_dist_to_own_centroid": dist_summary(own),
                 "known_test_dist_to_nearest_centroid": dist_summary(known_t["nc_dist"]),
                 "known_test_cosine_to_nearest_centroid_mean": float(known_t["nc_cos"].mean())}

    # ---------------- Grape Esca and Corn NLB deep analyses
    def deep(unknown, crop_prefix):
        u = df[df["class_name"] == unknown]
        ix = u.index.to_numpy()
        cen = Ete[ix].mean(0)
        rank_e = np.argsort(np.sqrt(((C - cen) ** 2).sum(1)))
        rank_c = np.argsort(-(l2n(cen[None]) @ Cn.T)[0])
        same_crop = [c for c in range(21) if classes[c].startswith(crop_prefix)]
        closest = min(same_crop, key=lambda c: np.sqrt(((C[c] - cen) ** 2).sum()))
        rows = []
        for c in same_crop + [int(rank_e[0])] if int(rank_e[0]) not in same_crop else same_crop:
            tr_c = Etr[tr["class_id"].to_numpy() == c]
            te_c = known_t[known_t["label"] == c]
            Xte_c = Ete[te_c.index.to_numpy()]
            # nearest-neighbour Euclidean distances to this class's TRAIN images
            def nn_d(Q):
                return np.sqrt(np.maximum((Q ** 2).sum(1)[:, None] - 2 * Q @ tr_c.T + (tr_c ** 2).sum(1), 0)).min(1)
            sub_u = Ete[rng.choice(ix, min(400, len(ix)), replace=False)]
            sub_t = tr_c[rng.choice(len(tr_c), min(400, len(tr_c)), replace=False)]
            pw = np.sqrt(np.maximum((sub_u ** 2).sum(1)[:, None] - 2 * sub_u @ sub_t.T + (sub_t ** 2).sum(1), 0)).ravel()
            pw_within = np.sqrt(np.maximum((Xte_c ** 2).sum(1)[:, None] - 2 * Xte_c @ sub_t.T + (sub_t ** 2).sum(1), 0)).ravel()
            rows.append({
                "unknown": unknown, "known_class": classes[c],
                "centroid_euclidean_unknown_to_known": float(np.sqrt(((C[c] - cen) ** 2).sum())),
                "centroid_cosine_unknown_to_known": float((l2n(cen[None]) @ Cn[c][:, None])[0, 0]),
                "rank_among_21_by_centroid_euclidean": int(np.where(rank_e == c)[0][0]) + 1,
                "mean_dist_unknown_images_to_known_centroid": float(eu[ix, c].mean()),
                "mean_dist_known_test_images_to_own_centroid": float(eu[te_c.index.to_numpy(), c].mean()),
                "mean_cosine_unknown_images_to_known_centroid": float(co[ix, c].mean()),
                "nn_dist_unknown_to_known_train_median": float(np.median(nn_d(Ete[ix]))),
                "nn_dist_known_test_to_own_train_median": float(np.median(nn_d(Xte_c))),
                "pairwise_unknown_vs_known_train_median": float(np.median(pw)),
                "pairwise_known_test_vs_own_train_median": float(np.median(pw_within)),
                "share_unknown_images_nearest_centroid_is_this": float((eu[ix].argmin(1) == c).mean()),
                "share_unknown_images_maha_nearest_is_this": float((maha_cls[ix] == c).mean()),
                "linear_probe_auroc_unknown_vs_known_train_DIAGNOSTIC": linear_probe(Ete[ix], tr_c, fo["seed"]),
                "maha_auroc_unknown_vs_this_class_test": ranking_metrics(te_c["maha"].to_numpy(), u["maha"].to_numpy())["auroc"],
                "msp_auroc_unknown_vs_this_class_test": ranking_metrics(te_c["u_msp"].to_numpy(), u["u_msp"].to_numpy())["auroc"],
                "energy_auroc_unknown_vs_this_class_test": ranking_metrics(te_c["energy"].to_numpy(), u["energy"].to_numpy())["auroc"]})
        return pd.DataFrame(rows), classes[closest], [classes[int(k)] for k in rank_e[:5]], [classes[int(k)] for k in rank_c[:5]]

    esca_t, esca_closest, esca_rank_e, esca_rank_c = deep("Grape___Esca_(Black_Measles)", "Grape___")
    nlb_t, nlb_closest, nlb_rank_e, nlb_rank_c = deep("Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___")

    # ---------------- correlations on the Experiment-B pool
    pool = df[df["role"].isin(["known", "unknown_near"])]
    cols = {"1-MSP": pool["u_msp"], "Energy": pool["energy"], "Mahalanobis": pool["maha"], "Nearest-centroid dist": pool["nc_dist"]}
    names = list(cols)
    corr = {"pearson": [[float(pearsonr(cols[a], cols[b])[0]) for b in names] for a in names],
            "spearman": [[float(spearmanr(cols[a], cols[b]).correlation) for b in names] for a in names], "names": names,
            "pool": "Experiment B: known PlantVillage test + hard unknowns"}

    # ---------------- disagreement table (hard unknowns)
    U = df[df["role"] == "unknown_near"]
    dec = lambda s, t: np.where(s <= t, "known", "unknown")  # noqa: E731
    dis = pd.DataFrame({"image_path": U["image_path"], "true_unknown": U["class_name"], "predicted_class": U["pred_class"],
                        "msp": U["msp"], "energy": U["energy"], "mahalanobis": U["maha"],
                        "msp_decision": dec(U["u_msp"], t_msp), "energy_decision": dec(U["energy"], t_en),
                        "mahalanobis_decision": dec(U["maha"], tau), "mahalanobis_nearest_class": U["maha_class"]})
    ok = lambda c: dis[c] == "unknown"  # noqa: E731
    dis["category"] = np.select(
        [~ok("msp_decision") & ~ok("energy_decision") & ~ok("mahalanobis_decision"),
         ~ok("msp_decision") & ~ok("energy_decision") & ok("mahalanobis_decision"),
         ~ok("msp_decision") & ok("mahalanobis_decision"), ~ok("energy_decision") & ok("mahalanobis_decision"),
         ok("msp_decision") & ok("energy_decision") & ok("mahalanobis_decision"),
         ok("energy_decision") & ~ok("mahalanobis_decision")],
        ["all_three_fail", "msp_and_energy_fail_maha_succeeds", "msp_fails_maha_succeeds", "energy_fails_maha_succeeds",
         "all_three_succeed", "energy_succeeds_maha_fails"], default="other")
    tdir = resolve(cfg["paths"]["tables"])
    dis.to_csv(tdir / "osr_method_disagreement.csv", index=False)
    cat = dis.groupby(["true_unknown", "category"]).size().unstack(fill_value=0)
    flags = {"msp_fails_maha_succeeds": int((~ok("msp_decision") & ok("mahalanobis_decision")).sum()),
             "energy_fails_maha_succeeds": int((~ok("energy_decision") & ok("mahalanobis_decision")).sum()),
             "maha_fails_energy_succeeds": int((ok("energy_decision") & ~ok("mahalanobis_decision")).sum()),
             "all_three_fail": int((~ok("msp_decision") & ~ok("energy_decision") & ~ok("mahalanobis_decision")).sum()),
             "all_three_succeed": int((ok("msp_decision") & ok("energy_decision") & ok("mahalanobis_decision")).sum()),
             "by_class": cat.to_dict(orient="index")}

    # ---------------- predictions + tables + json
    is_known = df["known_unknown"] == "known"
    pku = np.where(df["maha"] <= tau, "known", "unknown")
    pd.DataFrame({"image_path": df["image_path"], "dataset": df["dataset"], "role": df["role"],
                  "true_class": df["label_name"].where(is_known, df["class_name"]), "true_known_unknown": df["known_unknown"],
                  "predicted_class": np.where(pku == "known", df["pred_class"], "UNKNOWN"), "argmax_class": df["pred_class"],
                  "mahalanobis_nearest_class": df["maha_class"], "msp": df["msp"].map(lambda v: f"{v:.17g}"),
                  "energy": df["energy"].map(lambda v: f"{v:.17g}"), "mahalanobis": df["maha"].map(lambda v: f"{v:.17g}"),
                  "nearest_centroid_distance": df["nc_dist"], "mahalanobis_threshold": tau, "predicted_known_unknown": pku,
                  "correct_osr_decision": pku == df["known_unknown"], "strict_test": df["strict_test"]}).to_csv(
        resolve(cfg["paths"]["results"]) / "predictions" / "osr_mahalanobis_predictions.csv", index=False)
    prox_out = prox.rename(columns={"nearest_known_by_images": "nearest_known", "dist_to_nearest_centroid_mean": "distance_mean",
                                    "cosine_to_nearest_centroid_mean": "cosine_similarity_mean"})
    write_table(prox_out.round(6), tdir / "unknown_feature_proximity")
    write_table(esca_t.round(6), tdir / "grape_esca_feature_distance")
    write_table(nlb_t.round(6), tdir / "corn_nlb_feature_distance")
    write_table(per_cls.round(6), tdir / "osr_mahalanobis_per_unknown_class")

    # ---------------- PCA (fit on TRAIN known only) + t-SNE (visualisation only)
    pca = PCA(n_components=fo["pca_components"], random_state=fo["seed"]).fit(Etr)
    P_tr, P_te, P_va = pca.transform(Etr), pca.transform(Ete), pca.transform(Eva)
    np.savez_compressed(emb_dir(cfg) / "pca50_projections.npz", train=P_tr.astype(np.float32), test=P_te.astype(np.float32),
                        val=P_va.astype(np.float32), explained_variance_ratio=pca.explained_variance_ratio_)
    from sklearn.manifold import TSNE
    ts = fo["tsne"]
    pick = []
    for c, d in df[df["role"].isin(["known", "unknown_near"])].groupby("class_name"):
        pick += list(rng.choice(d.index.to_numpy(), min(ts["per_class_sample"], len(d)), replace=False))
    pick = np.array(sorted(pick))
    Y = TSNE(n_components=2, perplexity=ts["perplexity"], max_iter=ts["max_iter"], random_state=ts["seed"],
             init="pca").fit_transform(P_te[pick])

    env = json.loads((emb_dir(cfg) / "extraction_checks.json").read_text())
    out = {"method": "Mahalanobis (tied Ledoit-Wolf covariance, min squared distance)", "threshold": thr, "protocol_check": checks,
           "extraction_checks": env, "mahalanobis_results": maha_res, "msp_results": msp_res, "energy_results": en_res,
           "bootstrap": boot, "per_unknown_class": per_cls.to_dict(orient="records"),
           "feature_proximity": prox.to_dict(orient="records"), "known_reference": known_ref,
           "grape_esca": {"table": esca_t.to_dict(orient="records"), "closest_same_crop_known": esca_closest,
                          "top5_known_by_centroid_euclidean": esca_rank_e, "top5_known_by_centroid_cosine": esca_rank_c},
           "corn_nlb": {"table": nlb_t.to_dict(orient="records"), "closest_same_crop_known": nlb_closest,
                        "top5_known_by_centroid_euclidean": nlb_rank_e, "top5_known_by_centroid_cosine": nlb_rank_c},
           "correlations": corr, "disagreement": flags,
           "pca": {"fit_on": "known TRAINING embeddings only", "n_components": fo["pca_components"],
                   "explained_variance_ratio_first10": pca.explained_variance_ratio_[:10].tolist(),
                   "explained_variance_total": float(pca.explained_variance_ratio_.sum())},
           "tsne": {**ts, "n_points": int(len(pick)), "note": "visualisation only; coordinates not used for any decision"},
           "umap": "not used (umap-learn not installed; not added as a dependency)",
           "thresholds_used": {"msp_u": t_msp, "energy": t_en, "mahalanobis": tau},
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (resolve(cfg["paths"]["metrics"]) / "mahalanobis_results.json").write_text(json.dumps(out, indent=2, default=float))

    from src.visualization import embedding_plots as EP
    figs = resolve(cfg["paths"]["figures"]) / "embeddings"
    EP.pca_plot(df, P_te, P_va, va, pca, figs / "pca_known_unknown.png", near)
    EP.tsne_plot(df.loc[pick].reset_index(drop=True), Y, figs / "tsne_known_unknown.png", near, ts)
    EP.centroid_distance_plot(df, figs / "nearest_centroid_distance.png")
    EP.correlation_plot(corr, figs / "osr_score_correlation.png")
    EP.method_comparison(per_cls, msp_res, en_res, maha_res, boot, figs / "msp_energy_mahalanobis.png")
    EP.esca_feature_plot(df, P_te, near, figs / "grape_esca_feature_space.png", esca_t, nlb_t)
    for k in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown"):
        r = maha_res[k]
        print(f"{k:24s} MAHA AUROC {r['auroc']:.4f} AUPR {r['aupr_out']:.4f} FPR95 {r['fpr_at_95tpr']:.4f} det {r['unknown_recall']:.4f} "
              f"F1 {r['unknown_f1']:.4f} acc {r['known_acceptance_rate']:.4f} | energy {en_res[k]['auroc']:.4f} msp {msp_res[k]['auroc']:.4f}")
    print(per_cls[per_cls["experiment"] == "B_hard_same_crop"][["unknown_class", "msp_auroc", "energy_auroc", "mahalanobis_auroc",
                                                               "msp_detection", "energy_detection", "mahalanobis_detection"]].to_string(index=False))
    print(prox[prox["role"] == "unknown_near"][["unknown", "nearest_known_by_images", "share_images_nearest_to_it",
                                               "dist_to_nearest_centroid_mean", "cosine_to_nearest_centroid_mean"]].to_string(index=False))
    print("known reference:", known_ref)
    print("flags:", {k: v for k, v in flags.items() if k != "by_class"})
