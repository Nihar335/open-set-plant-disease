"""Stage 2 — build the frozen open-set protocol: roles, leakage-safe splits, manifests.

Inputs : config.yaml (osr_protocol), data/processed/inventory.csv (scripts/inspect_dataset.py),
         optionally data/processed/embedding_pairs.csv (scripts/find_near_duplicates.py)
Outputs: data/splits/manifest_all.csv          every image, its role/split, or why it is excluded
         data/splits/{train,val,test}.csv       shared splits (one model serves all experiments)
         data/splits/experiment_{A,B,C}_{train,val,test}.csv
         data/splits/class_mapping.yaml         explicit class ids, roles, counts per split
         data/splits/PROTOCOL_LOCK.json         sha256 of every manifest + the protocol config
         results/tables/dataset/split_*.csv|md, results/figures/dataset/split_*.png

No image file is copied, moved or deleted. Usage:  python scripts/prepare_dataset.py
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.splitting import assign_splits, split_groups  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.tables import write_table  # noqa: E402

MANIFEST_COLS = ["image_path", "dataset", "crop", "disease", "class_name", "class_id", "known_unknown",
                 "role", "split", "leaf_group_id", "split_group_id", "experiments", "eval_subset",
                 "pv_equivalent", "source_split", "md5", "excluded", "exclusion_reason", "per_class_metrics",
                 "strict_test"]
EXPERIMENTS = {
    "A": "Easy open set: known PlantVillage test vs unseen crops (A1) and non-plant textures (A2)",
    "B": "Hard same-crop open set: known PlantVillage test vs unseen diseases on known crops",
    "C": "Cross-domain open set: PlantDoc field images (known-field vs unknown-field), model trained on PlantVillage only",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_protocol(op: dict, pv: pd.DataFrame) -> dict:
    """Every PlantVillage class gets exactly one role; every named class exists on disk."""
    classes = set(pv["class_name"])
    crop_of = pv.groupby("class_name")["crop"].first()
    roles = {c: "known" for c in op["known_classes"]}
    for c in op["unknown_near"]:
        roles.setdefault(c, "unknown_near") if c not in roles else sys.exit(f"{c} has two roles")
    for c in op["unknown_val"]:
        roles.setdefault(c, "unknown_val") if c not in roles else sys.exit(f"{c} has two roles")
    for c in classes:
        if crop_of[c] in op["unknown_far_crops"]:
            if c in roles:
                sys.exit(f"{c}: crop listed as far-unknown but class already has role {roles[c]}")
            roles[c] = "unknown_far_crop"
    missing = set(roles) - classes
    unassigned = classes - set(roles)
    if missing or unassigned:
        sys.exit(f"Protocol/class mismatch. Not on disk: {sorted(missing)}; no role: {sorted(unassigned)}")
    known_crops = set(crop_of[op["known_classes"]])
    for c in op["unknown_near"]:
        if crop_of[c] not in known_crops:
            sys.exit(f"near-unknown {c} is not on a known crop")
        for r in op["unknown_near"][c]["resembles"]:
            if r not in op["known_classes"]:
                sys.exit(f"{c} resembles {r}, which is not a known class")
    if set(op["unknown_far_crops"]) & known_crops:
        sys.exit("a far-unknown crop is also a known crop")
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    op = cfg["osr_protocol"]
    rng = np.random.default_rng(op["seed"])
    proc, out = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["splits"])
    out.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(proc / "inventory.csv", low_memory=False, keep_default_na=False)
    inv["ok"] = inv["ok"].astype(str).str.lower().eq("true")
    pv_mask, pd_mask, dtd_mask = (inv["dataset"] == d for d in ("plantvillage", "plantdoc", "dtd"))
    roles = validate_protocol(op, inv[pv_mask])
    class_id = {c: i for i, c in enumerate(op["known_classes"])}

    m = pd.DataFrame(index=inv.index)
    m["image_path"] = inv["rel_path"]
    for c in ("dataset", "crop", "disease", "class_name", "source_split", "md5"):
        m[c] = inv[c]
    m["leaf_group_id"] = inv["leaf_group"]
    m["pv_equivalent"] = np.where(pv_mask, inv["class_name"], inv.get("pv_equivalent", ""))
    m["role"], m["split"], m["excluded"], m["exclusion_reason"] = "", "", False, ""
    m["experiments"], m["eval_subset"], m["per_class_metrics"] = "", "", True

    def exclude(mask, reason):
        new = mask & ~m["excluded"]
        m.loc[new, "excluded"] = True
        m.loc[new, "exclusion_reason"] = reason
        m.loc[new, "split"] = "excluded"

    exclude(~inv["ok"], "corrupt_or_unreadable")
    for path, reason in op.get("plantvillage_exclude", {}).items():
        if not (m["image_path"] == path).any():
            sys.exit(f"plantvillage_exclude path not in inventory: {path}")
        exclude(m["image_path"] == path, f"plantvillage_{reason.split(':')[0]}")

    # ---------------- PlantVillage
    m.loc[pv_mask, "role"] = inv.loc[pv_mask, "class_name"].map(roles)
    known = pv_mask & (m["role"] == "known") & ~m["excluded"]
    kdf = inv[known].copy()
    # extra "keep together" evidence beyond leaf_group + strong hash duplicates
    links = []
    merge_cos = cfg.get("near_duplicates", {}).get("merge_cosine")
    emb_file, sift_file = proc / "embedding_pairs.csv", proc / "same_leaf_pairs.csv"
    merge_unmapped = cfg.get("near_duplicates", {}).get("merge_cosine_unmapped")
    if emb_file.exists() and (merge_cos is not None or merge_unmapped is not None):
        e = pd.read_csv(emb_file, usecols=["i", "j", "score", "class_i", "class_j"])
        if merge_cos is not None:
            links.append(e[e["score"] >= merge_cos][["i", "j"]])
        if merge_unmapped is not None:
            unmapped = inv["leaf_match"].str.startswith("fallback").to_numpy()
            rule = (e["class_i"] == e["class_j"]) & (e["score"] >= merge_unmapped) &                 (unmapped[e["i"].to_numpy()] | unmapped[e["j"].to_numpy()])
            links.append(e[rule][["i", "j"]])
    if sift_file.exists():                                   # Level-4b geometrically verified same leaf
        links.append(pd.read_csv(sift_file, usecols=["i", "j"]))
    links = pd.concat(links, ignore_index=True) if links else None
    kdf["split_group_id"] = split_groups(kdf, links)
    m.loc[known, "split_group_id"] = kdf["split_group_id"]
    m.loc[known, "split"] = assign_splits(kdf, op["known_split_fractions"], op["seed"])
    m.loc[pv_mask & (m["role"] == "unknown_val") & ~m["excluded"], "split"] = "val"
    m.loc[pv_mask & m["role"].isin(["unknown_near", "unknown_far_crop"]) & ~m["excluded"], "split"] = "test"
    m.loc[known & (m["split"] == "test"), "experiments"] = "A;B"
    m.loc[pv_mask & (m["role"] == "unknown_near"), "experiments"] = "B"
    far = pv_mask & (m["role"] == "unknown_far_crop") & ~m["excluded"]
    m.loc[far, "experiments"] = "A"
    for c, idx in m[far].groupby("class_name").groups.items():
        take = rng.choice(np.array(idx), size=min(op["far_balanced_per_class"], len(idx)), replace=False)
        m.loc[take, "eval_subset"] = "far_balanced"

    # ---------------- DTD (non-plant far unknowns, partition-1 test list)
    dtd_dir = resolve(cfg["datasets"]["dtd"]["dir"])
    test_list = {f"{resolve(cfg['datasets']['dtd']['dir']).relative_to(ROOT).as_posix()}/images/{l.strip()}"
                 for l in (dtd_dir / op["dtd_split_file"]).read_text().splitlines() if l.strip()}
    m.loc[dtd_mask, "role"] = "unknown_far_texture"
    exclude(dtd_mask & ~m["image_path"].isin(test_list), f"dtd_not_in_{Path(op['dtd_split_file']).name}")
    m.loc[dtd_mask & ~m["excluded"], ["split", "experiments"]] = ["test", "A"]

    # ---------------- PlantDoc (field domain, test only)
    pdc = op["plantdoc"]
    pv_role = inv.loc[pd_mask, "pv_equivalent"].map(roles)
    m.loc[pd_mask, "role"] = pv_role.map({"known": "known_field", "unknown_near": "unknown_near_field",
                                          "unknown_far_crop": "unknown_far_field", "unknown_val": "unknown_val"})
    exclude(pd_mask & (m["role"].isin(pdc["exclude_roles"])),
            "plantdoc_class_reserved_for_validation_unknowns")
    clusters = inv.loc[pd_mask].groupby("dup_cluster")["class_name"].nunique()
    if pdc["drop_label_conflicts"]:
        conflict = inv["dup_cluster"].isin(clusters[clusters > 1].index) & pd_mask
        exclude(conflict, "plantdoc_label_conflict_same_image_in_two_classes")
    if pdc["dedupe_strong_duplicates"]:
        live = pd_mask & ~m["excluded"]
        first = inv[live].sort_values("rel_path").groupby("dup_cluster").head(1).index
        exclude(live & ~inv.index.isin(first), "plantdoc_duplicate_copy_one_kept")
    # a PlantDoc image identical to any PlantVillage image would contaminate the field test
    pv_clusters = set(inv.loc[pv_mask, "dup_cluster"])
    exclude(pd_mask & inv["dup_cluster"].isin(pv_clusters), "plantdoc_duplicate_of_plantvillage_image")
    live = pd_mask & ~m["excluded"]
    m.loc[live, ["split", "experiments"]] = ["test", "C"]
    small = m[live].groupby("class_name").size()
    small = small[small < pdc["min_images_for_per_class_metrics"]].index
    m.loc[live & m["class_name"].isin(small), "per_class_metrics"] = False

    # ---------------- strict test subset: test images with NO candidate partner (any method) in train/val
    cand = [pd.read_csv(proc / "duplicate_pairs.csv", usecols=["path_i", "path_j"])]
    if emb_file.exists():
        cand.append(pd.read_csv(emb_file, usecols=["path_i", "path_j"]))
    cand = pd.concat(cand, ignore_index=True)
    split_now = dict(zip(m["image_path"], m["split"]))
    si, sj = cand["path_i"].map(split_now), cand["path_j"].map(split_now)
    flagged = set(cand.loc[(si == "test") & sj.isin(["train", "val"]), "path_i"]) |         set(cand.loc[(sj == "test") & si.isin(["train", "val"]), "path_j"])
    m["strict_test"] = (m["split"] == "test") & ~m["image_path"].isin(flagged)

    # ---------------- ids / known-unknown
    label_cls = m["pv_equivalent"].where(m["dataset"] == "plantdoc", m["class_name"])
    is_known = m["role"].isin(["known", "known_field"])
    m["class_id"] = np.where(is_known, label_cls.map(class_id).fillna(-1), -1).astype(int)
    m["known_unknown"] = np.where(is_known, "known", "unknown")
    m.loc[m["dataset"] != "plantvillage", "split_group_id"] = m["image_path"].map(
        lambda p: "img_" + hashlib.md5(p.encode()).hexdigest()[:12])
    m["split_group_id"] = m["split_group_id"].fillna("")
    m = m[MANIFEST_COLS]

    # ---------------- write manifests
    m.to_csv(out / "manifest_all.csv", index=False)
    used = m[~m["excluded"]]
    shared = {"train": used[used["split"] == "train"], "val": used[used["split"] == "val"],
              "test": used[used["split"] == "test"]}
    for s, d in shared.items():
        d.to_csv(out / f"{s}.csv", index=False)
    for e in EXPERIMENTS:
        shared["train"].to_csv(out / f"experiment_{e}_train.csv", index=False)
        shared["val"].to_csv(out / f"experiment_{e}_val.csv", index=False)
        t = shared["test"][shared["test"]["experiments"].str.split(";").apply(lambda x: e in x)]
        t.to_csv(out / f"experiment_{e}_test.csv", index=False)

    # ---------------- class mapping
    def counts(frame, value):
        d = frame[frame["class_name"] == value]
        return {s: int((d["split"] == s).sum()) for s in ("train", "val", "test")}

    pv_used = used[used["dataset"] == "plantvillage"]
    near_cfg = op["unknown_near"]
    mapping = {
        "protocol_version": op["version"], "seed": op["seed"],
        "class_id_rule": "index in config.yaml osr_protocol.known_classes; unknown images have class_id -1",
        "known_classes": [{"class_id": class_id[c], "class_name": c, "dataset": "PlantVillage",
                           "crop": c.split("___")[0], "disease": c.split("___")[1],
                           "images": int((pv_used["class_name"] == c).sum()),
                           "split_counts": counts(pv_used, c)} for c in op["known_classes"]],
        "unknown_classes": {
            "unknown_near (Experiment B, test only)": [
                {"class_name": c, "dataset": "PlantVillage", "crop": c.split("___")[0],
                 "disease": c.split("___")[1], "images": int((pv_used["class_name"] == c).sum()),
                 "resembles_known": near_cfg[c]["resembles"], "why_hard": near_cfg[c]["why"]} for c in near_cfg],
            "unknown_far_crop (Experiment A1, test only)": [
                {"class_name": c, "dataset": "PlantVillage", "crop": c.split("___")[0],
                 "images": int((pv_used["class_name"] == c).sum()),
                 "balanced_subset": int(((pv_used["class_name"] == c) & (pv_used["eval_subset"] == "far_balanced")).sum())}
                for c in sorted(k for k, v in roles.items() if v == "unknown_far_crop")],
            "unknown_far_texture (Experiment A2, test only)": {
                "dataset": "DTD", "images": int((used["role"] == "unknown_far_texture").sum()),
                "textures": int(used.loc[used["role"] == "unknown_far_texture", "class_name"].nunique())},
        },
        "validation_classes": {
            "known_val": "the val split of every known class",
            "unknown_val (validation only; never in test)": [
                {"class_name": c, "images": int((pv_used["class_name"] == c).sum())} for c in op["unknown_val"]]},
        "test_classes": {
            "A": ["known PlantVillage test split", "unknown_far_crop", "unknown_far_texture (DTD)"],
            "B": ["known PlantVillage test split", "unknown_near"],
            "C": ["PlantDoc known_field", "PlantDoc unknown_near_field", "PlantDoc unknown_far_field"]},
        "plantdoc_mapping": [
            {"plantdoc_class": c, "pv_equivalent": g["pv_equivalent"].iat[0], "role": g["role"].iat[0],
             "class_id": int(g["class_id"].iat[0]), "images_total": int(len(g)),
             "images_used": int((~g["excluded"]).sum()),
             "excluded": {k: int(v) for k, v in g.loc[g["excluded"], "exclusion_reason"].value_counts().items()},
             "per_class_metrics": bool(g.loc[~g["excluded"], "per_class_metrics"].all()) if (~g["excluded"]).any() else False,
             **({"caveat": op["plantdoc"]["caveats"][c]} if c in op["plantdoc"]["caveats"] else {})}
            for c, g in m[m["dataset"] == "plantdoc"].groupby("class_name")],
        "experiments": EXPERIMENTS,
    }
    (out / "class_mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True, width=120),
                                            encoding="utf-8")

    # ---------------- lock
    files = sorted(p for p in out.glob("*.csv")) + [out / "class_mapping.yaml"]
    lock = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_config_sha256": hashlib.sha256(json.dumps(op, sort_keys=True).encode()).hexdigest(),
            "inventory_sha256": sha256_file(proc / "inventory.csv"),
            "embedding_merge_cosine": merge_cos, "embedding_merge_cosine_unmapped": merge_unmapped,
            "same_leaf_pairs_sha256": sha256_file(sift_file) if sift_file.exists() else None,
            "files": {p.name: sha256_file(p) for p in files},
            "rule": "Changing any protocol file after the first test-set evaluation invalidates reported results."}
    (out / "PROTOCOL_LOCK.json").write_text(json.dumps(lock, indent=2))

    # ---------------- summary tables
    tables = resolve(cfg["paths"]["tables"]) / "dataset"
    st = used.groupby(["dataset", "role", "split"]).size().unstack(fill_value=0).reindex(
        columns=["train", "val", "test"], fill_value=0).reset_index()
    write_table(st, tables / "split_summary")
    kc = pv_used[pv_used["role"] == "known"].groupby(["class_name", "split"]).size().unstack(fill_value=0)[
        ["train", "val", "test"]]
    kc["total"] = kc.sum(1)
    for s in ("train", "val", "test"):
        kc[f"{s}_%"] = (100 * kc[s] / kc["total"]).round(1)
    kc["leaf_groups"] = pv_used[pv_used["role"] == "known"].groupby("class_name")["leaf_group_id"].nunique()
    write_table(kc.reset_index(), tables / "split_known_class_counts")
    ex = m[m["excluded"]].groupby(["dataset", "exclusion_reason"]).size().rename("images").reset_index()
    write_table(ex, tables / "split_exclusions")
    exp = pd.DataFrame([{"experiment": e, "known_test": int((pd.read_csv(out / f"experiment_{e}_test.csv")["known_unknown"] == "known").sum()),
                         "known_test_strict": int(((t := pd.read_csv(out / f"experiment_{e}_test.csv"))["known_unknown"].eq("known") & t["strict_test"]).sum()),
                         "unknown_test": int((pd.read_csv(out / f"experiment_{e}_test.csv")["known_unknown"] == "unknown").sum()),
                         "train": len(shared["train"]), "val": len(shared["val"]), "description": EXPERIMENTS[e]}
                        for e in EXPERIMENTS])
    write_table(exp, tables / "split_experiments")

    from src.visualization import dataset_plots as P
    from src.visualization.style import apply_style
    apply_style()
    figs = resolve(cfg["paths"]["figures"]) / "dataset"
    P.known_unknown_distribution(used, figs / "split_known_unknown_distribution.png")
    P.split_distribution(pv_used[pv_used["role"] == "known"], figs / "split_train_val_test_per_known_class.png")
    P.manifest_grid(pv_used[pv_used["role"] == "known"], figs / "split_samples_known_classes.png",
                    "Known classes (training vocabulary) — one image each", ncols=7)
    P.manifest_grid(used[used["role"].isin(["unknown_near", "unknown_val"])],
                    figs / "split_samples_unknown_near_and_val.png",
                    "Unknown classes — near (test) and validation-unknown", ncols=4, per_class=2)
    P.manifest_grid(used[used["role"] == "unknown_far_crop"], figs / "split_samples_unknown_far_crops.png",
                    "Unknown classes — unseen crops (test)", ncols=9)
    P.domain_comparison(used, figs / "split_plantvillage_vs_plantdoc.png")
    if links is not None and len(links):
        # audit figures for the Level-4 rule: merged pairs, and the most similar pairs that still cross splits
        split_of = dict(zip(m["image_path"], m["split"]))
        e = pd.read_csv(emb_file)
        merged = e[(e["score"] >= merge_cos) & (e["category"] == "PV: same class, different leaf group")]
        P.pair_examples(inv, merged.sample(min(6, len(merged)), random_state=0).rename(columns={"score": "hamming"}),
                        figs / "dup_audit_embedding_pairs_merged.png",
                        f"Kept in the same split: cross-leaf-group pairs, cosine >= {merge_cos}", split_of=split_of)
        e["si"], e["sj"] = e["path_i"].map(split_of), e["path_j"].map(split_of)
        rest = e[e["si"].isin(["train", "val", "test"]) & e["sj"].isin(["train", "val", "test"]) & (e["si"] != e["sj"])]
        P.pair_examples(inv, rest.nlargest(6, "score").rename(columns={"score": "hamming"}),
                        figs / "dup_audit_highest_remaining_cross_split.png",
                        "Most similar pairs that still cross splits (flagged for review)", split_of=split_of)

    print(st.to_string(index=False))
    print("\nexclusions:\n" + ex.to_string(index=False))
    print("\n" + exp.drop(columns="description").to_string(index=False))
    print(f"\nwrote manifests + class_mapping.yaml + PROTOCOL_LOCK.json -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
