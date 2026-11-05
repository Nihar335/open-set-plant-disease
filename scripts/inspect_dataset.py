"""Dataset inventory + duplicate audit (Stage 2, steps 2 & 4).

Reads every downloaded image once and produces (all counts computed from files):
  data/processed/inventory.csv              one row per image: integrity, size, format, hashes,
                                            maintainer leaf id, leaf_group, strong dup cluster
  data/processed/duplicate_pairs.csv        every flagged pair, any level, with method + score
  results/metrics/dataset_inventory.json    machine-readable inventory (PlantVillage, PlantDoc, DTD)
  results/tables/dataset/*.csv|md           human-readable tables
  results/figures/dataset/*.png             distributions, sizes, backgrounds, samples, duplicate examples

Duplicate levels (see DATASET_CARD.md for limitations):
  L1  identical file bytes (md5)                                   -> strong
  L2  identical decoded pixels (pixel_md5)                          -> strong
      identical normalised 16x16 thumbnail (thumb_md5)              -> weak, report only
  L3  identical orientation-invariant pHash AND dHash (D4 min)      -> strong (catches flips/rotations)
      pHash Hamming <= threshold (original orientation)             -> weak, report only
  L4  CNN-embedding cosine similarity                               -> scripts/find_near_duplicates.py

"Strong" pairs are merged into `dup_cluster`; the splitter keeps each cluster in one split.
Nothing is deleted here.

Usage:  python scripts/inspect_dataset.py [--reanalyze]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import inventory as inv_mod  # noqa: E402
from src.data.duplicates import UnionFind, describe_pairs, hex_to_bits, near_duplicate_pairs  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.imaging import analyze_image  # noqa: E402
from src.utils.tables import to_md, write_table  # noqa: E402
from src.visualization import dataset_plots as P  # noqa: E402
from src.visualization.style import apply_style  # noqa: E402

STATS_COLS = [c for c in analyze_image(("", 1)) if c != "rel_path"]  # column names from the worker
HASH_COLS = ["md5", "pixel_md5", "thumb_md5", "phash", "dhash", "phash_d4", "dhash_d4"]


# ----------------------------------------------------------------------------- inventory + pixels
def build_inventory(cfg: dict) -> tuple[pd.DataFrame, dict]:
    frames, extras = [], {}
    for name, fn in [("plantvillage", inv_mod.plantvillage_inventory),
                     ("plantdoc", inv_mod.plantdoc_inventory), ("dtd", inv_mod.dtd_inventory)]:
        if not (resolve(cfg["datasets"][name]["dir"]) / ".download_complete.json").exists():
            print(f"  [skip] {name}: not downloaded")
            continue
        df, extra = fn(cfg)
        frames.append(df)
        extras[name] = extra
        print(f"  {name}: {len(df):,} image files")
    return pd.concat(frames, ignore_index=True), extras


def analyze_all(inv: pd.DataFrame, cfg: dict, cache: Path, reanalyze: bool) -> pd.DataFrame:
    cached = pd.DataFrame()
    if cache.exists() and not reanalyze:
        cached = pd.read_csv(cache, dtype={c: str for c in HASH_COLS + ["error", "mode", "format"]},
                             keep_default_na=False, na_values=[""])
        cached = cached[cached["rel_path"].isin(inv["rel_path"])]
        if not set(STATS_COLS) <= set(cached.columns):     # worker gained new fields -> recompute
            cached = pd.DataFrame()
    todo = inv.loc[~inv["rel_path"].isin(cached.get("rel_path", [])), "rel_path"].tolist()
    if todo:
        ic = cfg["inspection"]
        jobs = [(str(ROOT / p), ic["border_px"]) for p in todo]
        with Pool(ic["num_workers"]) as pool:
            res = list(tqdm(pool.imap(analyze_image, jobs, chunksize=64), total=len(jobs), desc="  decoding"))
        new = pd.DataFrame(res)
        new.insert(0, "rel_path", todo)
        cached = pd.concat([cached, new], ignore_index=True)
    return inv.merge(cached[["rel_path"] + STATS_COLS], on="rel_path", how="left")


# ----------------------------------------------------------------------------- duplicates
def equal_key_pairs(inv: pd.DataFrame, keys: list[str], max_all_pairs: int = 50) -> pd.DataFrame:
    """Pairs of OK images whose `keys` columns are all equal (all pairs for small groups)."""
    ok = inv[inv["ok"]]
    rows = []
    for _, idx in ok.groupby(keys).groups.items():
        idx = sorted(idx)
        if len(idx) < 2:
            continue
        it = combinations(idx, 2) if len(idx) <= max_all_pairs else ((idx[0], j) for j in idx[1:])
        rows += [(i, j) for i, j in it]
    return pd.DataFrame(rows, columns=["i", "j"]).assign(score=1.0)


def find_duplicates(inv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ic = cfg["inspection"]
    parts = [
        describe_pairs(equal_key_pairs(inv, ["md5"]), inv, "L1_md5", "strong"),
        describe_pairs(equal_key_pairs(inv, ["pixel_md5"]), inv, "L2_pixel_md5", "strong"),
        describe_pairs(equal_key_pairs(inv, ["thumb_md5"]), inv, "L2_thumb16_md5", "weak"),
        describe_pairs(equal_key_pairs(inv, ["phash_d4", "dhash_d4"]), inv, "L3_phash+dhash_D4_equal", "strong"),
    ]
    ok_idx = np.flatnonzero(inv["ok"].to_numpy())
    near = near_duplicate_pairs(hex_to_bits(inv.loc[ok_idx, "phash"]), ic["phash_hamming_threshold"])
    near = pd.DataFrame({"i": ok_idx[near["i"]], "j": ok_idx[near["j"]], "score": near["hamming"]})
    parts.append(describe_pairs(near, inv, f"L3_phash_hamming<={ic['phash_hamming_threshold']}", "weak"))
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- tables
def _counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def dataset_block(d: pd.DataFrame, extra: dict, dups: pd.DataFrame) -> dict:
    """Common inventory fields for one dataset."""
    ok = d[d["ok"]]
    per_class = d.groupby("class_name").size()
    in_ds = dups[(dups["dataset_i"] == d["dataset"].iat[0]) & (dups["dataset_j"] == d["dataset"].iat[0])]
    return {
        "total_images": int(len(d)), "decoded_ok": int(len(ok)), "corrupt_or_unreadable": int((~d["ok"]).sum()),
        "corrupt_files": d.loc[~d["ok"], ["rel_path", "error"]].to_dict(orient="records"),
        "non_image_files": extra.get("non_image_files", []),
        "n_classes": int(per_class.size), "images_per_class": {k: int(v) for k, v in per_class.items()},
        "n_crops": int(d["crop"].nunique()), "images_per_crop": _counts(d["crop"]),
        "classes_per_crop": {k: sorted(v) for k, v in d.groupby("crop")["class_name"].unique().items()},
        "healthy_classes": sorted(d.loc[d["is_healthy"], "class_name"].unique()),
        "disease_classes": sorted(d.loc[~d["is_healthy"], "class_name"].unique()),
        "imbalance_max_over_min": round(float(per_class.max() / per_class.min()), 2),
        "formats": _counts(ok["format"]), "colour_modes": _counts(ok["mode"]),
        "file_extensions": _counts(d["file_name"].str.rsplit(".", n=1).str[-1].str.lower()),
        "dimensions": {"distinct_sizes": int(ok.groupby(["width", "height"]).ngroups),
                       "width": ok["width"].describe().round(1).to_dict(),
                       "height": ok["height"].describe().round(1).to_dict(),
                       "most_common": {f"{int(w)}x{int(h)}": int(n) for (w, h), n in
                                       ok.groupby(["width", "height"]).size().nlargest(5).items()}},
        "exif_rotated_images": int((ok["exif_orientation"].fillna(1).astype(int) != 1).sum()),
        "duplicate_files_md5_redundant_copies": int(ok.duplicated("md5").sum()),
        "exact_duplicate_pixels_redundant_copies": int(ok.duplicated("pixel_md5").sum()),
        "duplicate_pairs_within_dataset_by_method": {
            m: {"pairs": int(len(g)), "by_category": _counts(g["category"])} for m, g in in_ds.groupby("method")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--reanalyze", action="store_true", help="ignore the cached per-image statistics")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc, tables = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["tables"]) / "dataset"
    figs, metrics = resolve(cfg["paths"]["figures"]) / "dataset", resolve(cfg["paths"]["metrics"])
    for d in (proc, tables, figs, metrics):
        d.mkdir(parents=True, exist_ok=True)
    apply_style()

    print("[1/5] Building file inventory")
    inv, extras = build_inventory(cfg)

    print("[2/5] Decoding images (integrity, size, format, hashes, background stats)")
    inv = analyze_all(inv, cfg, proc / "image_stats_cache.csv", args.reanalyze)
    inv["ok"] = inv["ok"].astype(str).str.lower().eq("true")
    inv[["rel_path"] + STATS_COLS].to_csv(proc / "image_stats_cache.csv", index=False)
    inv[HASH_COLS] = inv[HASH_COLS].fillna("")

    print("[3/5] Duplicate search (L1-L3)")
    dups = find_duplicates(inv, cfg)
    dups.to_csv(proc / "duplicate_pairs.csv", index=False)
    uf = UnionFind(len(inv))
    strong = dups[dups["strength"] == "strong"]
    for i, j in zip(strong["i"], strong["j"]):
        uf.union(int(i), int(j))
    inv["dup_cluster"] = uf.labels()
    inv.to_csv(proc / "inventory.csv", index=False)

    print("[4/5] Inventory JSON + tables")
    pv, pdoc, dtd = (inv[inv["dataset"] == n] for n in ("plantvillage", "plantdoc", "dtd"))
    report: dict = {"generated_from": "scripts/inspect_dataset.py (all values computed from downloaded files)",
                    "provenance": json.loads((resolve(cfg["paths"]["raw"]) / "PROVENANCE.json").read_text())}
    summary_rows = []
    for name, d in (("plantvillage", pv), ("plantdoc", pdoc), ("dtd", dtd)):
        if len(d):
            report[name] = dataset_block(d, extras.get(name, {}), dups)
            b = report[name]
            summary_rows.append({"dataset": name, "images": b["total_images"], "corrupt": b["corrupt_or_unreadable"],
                                 "classes": b["n_classes"], "crops": b["n_crops"],
                                 "healthy_classes": len(b["healthy_classes"]),
                                 "min/class": min(b["images_per_class"].values()),
                                 "max/class": max(b["images_per_class"].values()),
                                 "imbalance": b["imbalance_max_over_min"],
                                 "distinct_sizes": b["dimensions"]["distinct_sizes"],
                                 "formats": ", ".join(f"{k}:{v}" for k, v in b["formats"].items()),
                                 "dup_files(md5)": b["duplicate_files_md5_redundant_copies"]})
    write_table(pd.DataFrame(summary_rows), tables / "dataset_summary")

    if len(pv):
        sizes = pv.groupby("leaf_group").size()
        spans = pv.groupby("leaf_group")["source_split"].nunique()
        report["plantvillage"]["leaf_grouping"] = {
            "algorithm": "maintainer HF loader (plant_village.py) leaf_id; group key = class_name||leaf_id",
            "leaf_id_resolution": _counts(pv["leaf_match"]),
            "images_without_valid_leaf_map_id": int(pv["leaf_match"].str.startswith("fallback").sum()),
            "unique_map_entry_naming_other_class": int(pv["leaf_map_other_class"].sum()),
            "n_physical_leaf_groups": int(sizes.size),
            "images_per_leaf_group": {"histogram": _counts(sizes.clip(upper=20).astype(str).replace("20", "20+")),
                                      "median": float(sizes.median()), "mean": round(float(sizes.mean()), 2),
                                      "max": int(sizes.max())},
            "official_hf_split_groups_spanning_train_and_test": int((spans > 1).sum()),
            "official_hf_split_images_in_spanning_groups": int(pv["leaf_group"].isin(spans[spans > 1].index).sum()),
            "files_with_copy_in_name": int(pv["name_has_copy"].sum()),
        }
        pv_cls = pv.groupby(["crop", "disease", "class_name"]).agg(
            images=("rel_path", "size"), leaf_groups=("leaf_group", "nunique"),
            no_leaf_map_id=("leaf_match", lambda s: int(s.str.startswith("fallback").sum())),
            official_train=("source_split", lambda s: int((s == "train").sum())),
            official_test=("source_split", lambda s: int((s == "test").sum()))).reset_index()
        write_table(pv_cls.sort_values(["crop", "disease"]), tables / "plantvillage_class_counts")
        crop_t = pv.groupby("crop").agg(classes=("class_name", "nunique"), images=("rel_path", "size"),
                                        leaf_groups=("leaf_group", "nunique")).reset_index()
        write_table(crop_t.sort_values("images", ascending=False), tables / "plantvillage_crop_counts")

    if len(pdoc):
        tt = pdoc.groupby(["class_name", "source_split"]).size().unstack(fill_value=0)
        report["plantdoc"]["source_train_test"] = {
            "images_per_split": _counts(pdoc["source_split"]),
            "classes_in_train_folder": sorted(tt.index[tt.get("train", 0) > 0]),
            "classes_in_test_folder": sorted(tt.index[tt.get("test", 0) > 0]),
            "classes_only_in_train_folder": sorted(tt.index[(tt.get("train", 0) > 0) & (tt.get("test", 0) == 0)]),
            "per_class": {c: {"train": int(r.get("train", 0)), "test": int(r.get("test", 0))} for c, r in tt.iterrows()},
            "renamed_for_windows_portability": len(report["provenance"]["plantdoc"].get("renamed_for_portability", {})),
        }
        pd_cls = pdoc.groupby(["crop", "disease", "class_name", "pv_equivalent"]).agg(
            images=("rel_path", "size"),
            train_folder=("source_split", lambda s: int((s == "train").sum())),
            test_folder=("source_split", lambda s: int((s == "test").sum())),
            median_W=("width", "median"), median_H=("height", "median")).reset_index()
        write_table(pd_cls.sort_values(["crop", "disease"]), tables / "plantdoc_class_counts")

    dup_t = (dups.groupby(["method", "strength", "category"]).agg(
        pairs=("i", "size"), images=("i", lambda s: len(set(s) | set(dups.loc[s.index, "j"])))).reset_index())
    write_table(dup_t, tables / "duplicate_summary")
    cl = inv.groupby("dup_cluster").size()
    multi = inv[inv["dup_cluster"].map(cl) > 1].groupby("dup_cluster")
    report["cross_dataset"] = {
        "duplicate_pairs_by_method_and_category": {
            m: _counts(g["category"]) for m, g in dups.groupby("method")},
        "strong_clusters": {"clusters_with_>1_image": int((cl > 1).sum()), "largest": int(cl.max()),
                            "spanning_multiple_classes": int((multi["class_name"].nunique() > 1).sum()),
                            "spanning_multiple_datasets": int((multi["dataset"].nunique() > 1).sum()),
                            "pv_spanning_multiple_leaf_groups": int(
                                (inv[inv["dataset"] == "plantvillage"].groupby("dup_cluster")["leaf_group"].nunique() > 1).sum())},
        "level4_embeddings": "see results/metrics/near_duplicates_embedding.json (scripts/find_near_duplicates.py)",
    }
    (metrics / "dataset_inventory.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("[5/5] Figures")
    if len(pv):
        P.class_distribution(pv, figs / "plantvillage_class_distribution.png")
        P.crop_distribution(pv, figs / "plantvillage_crop_distribution.png", "PlantVillage: images per crop")
        P.leaf_group_sizes(pv, figs / "plantvillage_leaf_group_sizes.png")
        P.sample_grid(pv, figs / "plantvillage_samples.png", "PlantVillage — one random image per class")
    if len(pdoc):
        P.plantdoc_distribution(pdoc, figs / "plantdoc_class_distribution.png")
        P.crop_distribution(pdoc, figs / "plantdoc_crop_distribution.png", "PlantDoc: images per crop")
        P.sample_grid(pdoc, figs / "plantdoc_samples.png", "PlantDoc — one random image per class")
    P.image_sizes(inv, figs / "image_dimensions.png")
    P.background_stats(inv, figs / "background_statistics.png")
    for old in figs.glob("near_dup_examples__*.png"):
        old.unlink()
    for (method, cat), sel in dups[dups["category"] != "dtd internal"].groupby(["method", "category"]):
        slug = "".join(ch if ch.isalnum() else "_" for ch in f"{method}__{cat}".lower()).strip("_")
        P.pair_examples(inv, sel.sample(min(6, len(sel)), random_state=0).rename(columns={"score": "hamming"}),
                        figs / f"dup_examples__{slug}.png", f"{method}: {cat} (random examples)")

    print("\n" + to_md(pd.DataFrame(summary_rows)))
    print("\n" + to_md(dup_t))
    print(f"\nWrote {(metrics / 'dataset_inventory.json').relative_to(ROOT)}, inventory, tables, figures")


if __name__ == "__main__":
    main()
