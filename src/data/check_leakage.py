"""Validate the whole dataset protocol. Ends with PASS or FAIL (exit code 0 / 1).

    python -m src.data.check_leakage                     # validate data/splits
    python -m src.data.check_leakage --splits-dir X      # validate another manifest folder
    python -m src.data.check_leakage --skip-md5          # skip re-hashing image bytes (faster)

Checks
  1  protocol lock         manifests unchanged since prepare_dataset.py froze them
  2  manifest integrity    schema, unique paths, shared splits == manifest, experiment files are subsets
  3  missing files         every referenced image exists
  4  corrupt images        every used image decoded OK during inspection; bytes still match recorded md5
  5  physical-leaf leakage no PlantVillage leaf group (class||maintainer leaf id) in >1 split
  6  split-group leakage   no leaf+duplicate component in >1 split
  7  exact-duplicate leak. no strong duplicate pair (md5 / pixels / orientation-invariant hashes) crosses splits
  8  near-duplicate report weak hash + CNN-embedding candidates that cross splits (WARN, for review)
  9  unknown-class leakage train/val-known contain only known classes; unknowns never in train
  10 label leakage         class ids follow the explicit mapping; unknown classes never map to a known id
  11 test contamination    no test image (path or bytes) also in train or val
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.config import PROJECT_ROOT, load_config, resolve

REQUIRED = ["image_path", "dataset", "crop", "disease", "class_name", "class_id", "known_unknown", "role", "split",
            "leaf_group_id", "split_group_id", "experiments", "md5", "excluded"]
USED_SPLITS = ("train", "val", "test")


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check: str, status: str, detail: str, **data):
        self.rows.append({"check": check, "status": status, "detail": detail, **({"data": data} if data else {})})
        mark = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}[status]
        print(f"  [{mark}] {check}: {detail}")

    @property
    def failed(self) -> bool:
        return any(r["status"] == "FAIL" for r in self.rows)


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--skip-md5", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config)
    op = cfg["osr_protocol"]
    sd = Path(args.splits_dir) if args.splits_dir else resolve(cfg["paths"]["splits"])
    proc = resolve(cfg["paths"]["processed"])
    R = Report()
    print(f"Validating protocol in {sd}\n")

    # 1 ------------------------------------------------------------------ lock
    lock_p = sd / "PROTOCOL_LOCK.json"
    if not lock_p.exists():
        R.add("1 protocol lock", "FAIL", "PROTOCOL_LOCK.json missing — run scripts/prepare_dataset.py")
    else:
        lock = json.loads(lock_p.read_text())
        changed = [f for f, h in lock["files"].items()
                   if not (sd / f).exists() or hashlib.sha256((sd / f).read_bytes()).hexdigest() != h]
        cfg_changed = hashlib.sha256(json.dumps(op, sort_keys=True).encode()).hexdigest() != lock["protocol_config_sha256"]
        if changed or cfg_changed:
            R.add("1 protocol lock", "FAIL", f"changed since freeze: files={changed}, protocol_config={cfg_changed}")
        else:
            R.add("1 protocol lock", "PASS", f"{len(lock['files'])} files match the lock created {lock['created_utc']}")

    # 2 ------------------------------------------------------------------ manifest integrity
    m = pd.read_csv(sd / "manifest_all.csv", low_memory=False, keep_default_na=False)
    m["excluded"] = m["excluded"].astype(str).str.lower().eq("true")
    m["class_id"] = m["class_id"].astype(int)
    problems = [f"missing columns {sorted(set(REQUIRED) - set(m.columns))}"] if set(REQUIRED) - set(m.columns) else []
    if m["image_path"].duplicated().any():
        problems.append(f"{int(m['image_path'].duplicated().sum())} duplicated image_path rows")
    used = m[~m["excluded"]]
    if set(used["split"]) - set(USED_SPLITS):
        problems.append(f"unexpected split values {set(used['split']) - set(USED_SPLITS)}")
    if (m.loc[m["excluded"], "split"] != "excluded").any() or (m.loc[m["excluded"], "exclusion_reason"] == "").any():
        problems.append("excluded rows without split=excluded or without a reason")
    shared = {}
    for s in USED_SPLITS:
        f = pd.read_csv(sd / f"{s}.csv", low_memory=False, keep_default_na=False)
        shared[s] = f
        if set(f["image_path"]) != set(used.loc[used["split"] == s, "image_path"]):
            problems.append(f"{s}.csv differs from manifest_all split={s}")
    for e in ("A", "B", "C"):
        for s in USED_SPLITS:
            f = pd.read_csv(sd / f"experiment_{e}_{s}.csv", low_memory=False, keep_default_na=False)
            if not set(f["image_path"]) <= set(shared[s]["image_path"]):
                problems.append(f"experiment_{e}_{s}.csv contains rows outside {s}.csv")
            if s == "test" and not f["experiments"].str.split(";").apply(lambda x: e in x).all():
                problems.append(f"experiment_{e}_test.csv has rows not tagged {e}")
    R.add("2 manifest integrity", "FAIL" if problems else "PASS",
          "; ".join(problems) if problems else
          f"{len(m):,} manifest rows, {len(used):,} used, {int(m['excluded'].sum()):,} excluded with reasons")

    # 3 ------------------------------------------------------------------ files exist
    missing = [p for p in used["image_path"] if not (PROJECT_ROOT / p).is_file()]
    R.add("3 missing files", "FAIL" if missing else "PASS",
          f"{len(missing)} missing, e.g. {missing[:3]}" if missing else f"all {len(used):,} referenced images exist")

    # 4 ------------------------------------------------------------------ corrupt
    inv = pd.read_csv(proc / "inventory.csv", low_memory=False, keep_default_na=False)
    inv["ok"] = inv["ok"].astype(str).str.lower().eq("true")
    ok = inv.set_index("rel_path")["ok"]
    bad = [p for p in used["image_path"] if not ok.get(p, False)]
    changed = []
    if not args.skip_md5:
        changed = [p for p, h in zip(used["image_path"], used["md5"])
                   if (PROJECT_ROOT / p).is_file() and md5_file(PROJECT_ROOT / p) != h]
    detail = (f"{len(bad)} not decodable, {len(changed)} changed on disk since inspection"
              + ("" if not args.skip_md5 else " (md5 re-check skipped)"))
    R.add("4 corrupt images", "FAIL" if bad or changed else "PASS", detail)

    # 5 ------------------------------------------------------------------ physical-leaf leakage
    pv = used[used["dataset"] == "plantvillage"]
    span = pv.groupby("leaf_group_id")["split"].nunique()
    leaks = span[span > 1]
    pvk = pv[pv["role"] == "known"]
    per_split = {s: int(pvk.loc[pvk["split"] == s, "leaf_group_id"].nunique()) for s in USED_SPLITS}
    all_split = {s: int(pv.loc[pv["split"] == s, "leaf_group_id"].nunique()) for s in USED_SPLITS}
    pairs = {f"{a}+{b}": int(sum(1 for g in leaks.index
                                 if {a, b} <= set(pv.loc[pv["leaf_group_id"] == g, "split"])))
             for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    R.add("5 physical-leaf leakage", "FAIL" if len(leaks) else "PASS",
          f"known-class leaf groups: train {per_split['train']:,}, val {per_split['val']:,}, test {per_split['test']:,}; "
          f"(all PlantVillage incl. unknowns: {all_split}); cross-split leaf groups: {len(leaks)} {pairs}",
          known_groups_per_split=per_split, all_pv_groups_per_split=all_split, cross_split=int(len(leaks)))

    # 6 ------------------------------------------------------------------ split-group leakage
    k = pv[pv["role"] == "known"]
    sg = k.groupby("split_group_id")["split"].nunique()
    R.add("6 split-group leakage", "FAIL" if (sg > 1).any() else "PASS",
          f"{int((sg > 1).sum())} of {len(sg):,} leaf+duplicate components span splits")

    # 7/8 --------------------------------------------------------------- duplicate leakage
    split_of = used.set_index("image_path")["split"]
    cand = [pd.read_csv(proc / "duplicate_pairs.csv", low_memory=False)]
    for f in ("embedding_pairs.csv", "same_leaf_pairs.csv"):     # same_leaf_pairs are strength=strong
        if (proc / f).exists():
            cand.append(pd.read_csv(proc / f, low_memory=False))
    dp = pd.concat(cand, ignore_index=True)
    dp["split_i"], dp["split_j"] = dp["path_i"].map(split_of), dp["path_j"].map(split_of)
    cross = dp[dp["split_i"].notna() & dp["split_j"].notna() & (dp["split_i"] != dp["split_j"])].copy()
    cross["crosses"] = [f"{min(a, b)}<->{max(a, b)}" for a, b in zip(cross["split_i"], cross["split_j"])]
    merge_cos = cfg.get("near_duplicates", {}).get("merge_cosine")
    rule = cross["method"].str.startswith("L4") & (cross["score"] >= merge_cos) if merge_cos is not None         else pd.Series(False, index=cross.index)
    mu = cfg.get("near_duplicates", {}).get("merge_cosine_unmapped")
    if mu is not None:
        unm = set(inv.loc[inv["leaf_match"].str.startswith("fallback"), "rel_path"])
        rule |= cross["method"].str.startswith("L4") & (cross["score"] >= mu) &             (cross["class_i"] == cross["class_j"]) & (cross["path_i"].isin(unm) | cross["path_j"].isin(unm))
    strong = cross[(cross["strength"] == "strong") | rule]
    R.add("7 exact-duplicate leakage", "FAIL" if len(strong) else "PASS",
          f"{len(strong)} strong duplicate pairs (md5 / pixel / orientation-invariant hash"
          f"{'' if merge_cos is None else f' / embedding cos>={merge_cos} (>={mu} if unmapped) merge rules'} / SIFT-verified same leaf) cross splits"
          + (f": {strong['crosses'].value_counts().to_dict()}" if len(strong) else ""))
    weak = cross.drop(strong.index)
    out_t = resolve(cfg["paths"]["tables"]) / "dataset"
    out_t.mkdir(parents=True, exist_ok=True)
    weak_cols = ["method", "score", "crosses", "category", "path_i", "class_i", "split_i", "path_j", "class_j", "split_j"]
    weak.sort_values(["method", "score"], ascending=[True, False])[weak_cols].to_csv(
        out_t / "leakage_cross_split_candidates.csv", index=False)
    summary = {f"{meth}": g["crosses"].value_counts().to_dict() for meth, g in weak.groupby("method")}
    emb = weak[weak["method"].str.startswith("L4")]
    hi = emb[emb["score"] >= 0.98]
    R.add("8 near-duplicate report", "WARN" if len(weak) else "PASS",
          f"{len(weak):,} weak/candidate cross-split pairs for review (not proof of duplication): {summary}; "
          f"embedding cos>=0.98: {len(hi)} -> results/tables/dataset/leakage_cross_split_candidates.csv",
          by_method=summary, embedding_cos_ge_0_98=int(len(hi)))

    # 9 ------------------------------------------------------------------ unknown-class leakage
    known_set = set(op["known_classes"])
    unknown_named = set(op["unknown_near"]) | set(op["unknown_val"])
    tr = used[used["split"] == "train"]
    va = used[used["split"] == "val"]
    te = used[used["split"] == "test"]
    issues = []
    if (tr["known_unknown"] != "known").any() or (tr["role"] != "known").any():
        issues.append("train contains non-known rows")
    if not set(tr["class_name"]) <= known_set:
        issues.append(f"train classes outside known list: {sorted(set(tr['class_name']) - known_set)}")
    if set(tr["dataset"]) != {"plantvillage"} or set(va["dataset"]) != {"plantvillage"}:
        issues.append(f"train/val datasets {set(tr['dataset'])}/{set(va['dataset'])} (PlantDoc/DTD must be test-only)")
    if set(tr["class_name"]) & unknown_named:
        issues.append("an unknown class appears in train")
    va_unknown_roles = set(va.loc[va["known_unknown"] == "unknown", "role"])
    if va_unknown_roles - {"unknown_val"}:
        issues.append(f"val contains test-unknown roles {va_unknown_roles - {'unknown_val'}}")
    if set(te["role"]) & {"unknown_val"} or set(te.loc[te["dataset"] == "plantdoc", "pv_equivalent"]) & set(op["unknown_val"]):
        issues.append("validation-unknown classes appear in test")
    far_crops = set(op["unknown_far_crops"])
    if set(tr["crop"]) & far_crops:
        issues.append("an unseen (far-unknown) crop appears in train")
    missing_known = known_set - set(tr["class_name"])
    if missing_known:
        issues.append(f"known classes without training images: {sorted(missing_known)}")
    for s, d in (("val", va), ("test", te)):
        kk = d[(d["dataset"] == "plantvillage") & (d["role"] == "known")]
        lacking = known_set - set(kk["class_name"])
        if lacking:
            issues.append(f"known classes missing from {s}: {sorted(lacking)}")
    R.add("9 unknown-class leakage", "FAIL" if issues else "PASS",
          "; ".join(issues) if issues else
          f"train = {tr['class_name'].nunique()} known classes only; unknowns: val-unknown "
          f"{int((va['known_unknown'] == 'unknown').sum()):,}, test-unknown {int((te['known_unknown'] == 'unknown').sum()):,}")

    # 10 ----------------------------------------------------------------- label leakage
    issues = []
    cid = {c: i for i, c in enumerate(op["known_classes"])}
    lab = np.where(used["dataset"] == "plantdoc", used["pv_equivalent"], used["class_name"])
    expect = np.where(used["known_unknown"] == "known", pd.Series(lab).map(cid).fillna(-99).to_numpy(), -1)
    wrong = int((used["class_id"].to_numpy() != expect).sum())
    if wrong:
        issues.append(f"{wrong} rows whose class_id disagrees with the explicit mapping")
    if (used.loc[used["known_unknown"] == "unknown", "class_id"] != -1).any():
        issues.append("unknown rows with a known class id")
    unk_lab = set(pd.Series(lab)[(used["known_unknown"] == "unknown").to_numpy()])
    if unk_lab & known_set:
        issues.append(f"unknown rows labelled with known classes: {sorted(unk_lab & known_set)}")
    if set(op["unknown_near"]) & known_set or set(op["unknown_val"]) & known_set:
        issues.append("a class is both known and unknown in config")
    cm = yaml.safe_load((sd / "class_mapping.yaml").read_text(encoding="utf-8"))
    if {c["class_name"]: c["class_id"] for c in cm["known_classes"]} != cid:
        issues.append("class_mapping.yaml ids differ from config order")
    R.add("10 label leakage", "FAIL" if issues else "PASS",
          "; ".join(issues) if issues else f"{len(cid)} known ids 0..{len(cid) - 1} match config; all unknown rows have id -1")

    # 11 ----------------------------------------------------------------- test contamination
    tv_paths = set(tr["image_path"]) | set(va["image_path"])
    tv_md5 = set(tr["md5"]) | set(va["md5"])
    p_overlap = set(te["image_path"]) & tv_paths
    h_overlap = set(te["md5"]) & tv_md5
    R.add("11 test contamination", "FAIL" if p_overlap or h_overlap else "PASS",
          f"test & (train | val): {len(p_overlap)} paths, {len(h_overlap)} identical files")

    verdict = "FAIL" if R.failed else "PASS"
    out = resolve(cfg["paths"]["metrics"]) / "leakage_report.json"
    if not args.splits_dir:
        out.write_text(json.dumps({"verdict": verdict, "checks": R.rows}, indent=2, default=str))
    print(f"\n{verdict}" + ("" if verdict == "PASS" else
                           " — fix every [FAIL] above before any training. Details: results/metrics/leakage_report.json"))
    return 1 if R.failed else 0


if __name__ == "__main__":
    sys.exit(main())
