"""Unified crop coverage registry for Stage 11 (generated; never hand-edited).

For every plant in stage11/data/plants.yaml it records the names, the trained image classes and their image counts per
source and split, data provenance / licences, the image-model coverage level (KB.model_coverage), the field-evaluation
status (from stage11/results/multicrop_eval.json when it exists), the knowledge-base coverage (Stage 11 database and
the Stage 10 sourced KB, read-only) and known limitations.

Support categories (the four the project distinguishes):
  trained      - image classifier trained on healthy AND diseased photos of this crop (level "diagnosable")
  partial      - classifier covers only some conditions of this crop, or only a related crop ("limited" / "related")
  symptom_only - no trained image classes; database diseases are matched by detected symptoms
  unsupported  - nothing in the database for this plant

Run:  python -P stage11/coverage.py            -> stage11/results/crop_coverage.json + CROP_COVERAGE.md
      python -P stage11/coverage.py --check    (exit 1 if the files are out of date)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.kb import load_kb  # noqa: E402

RESULTS = ROOT / "stage11" / "results"
OUT_JSON = RESULTS / "crop_coverage.json"
OUT_MD = RESULTS / "CROP_COVERAGE.md"
STAGE10_REGISTRY = ROOT / "stage10" / "knowledge_base" / "coverage_registry.json"

PROVENANCE = {
    "plantvillage": {"name": "PlantVillage (lab photos, plain background)", "licence": "CC BY-SA 3.0 (HF card; verify)",
                     "domain": "lab"},
    "plantdoc": {"name": "PlantDoc (web photos)", "licence": "CC BY 4.0", "domain": "field/web"},
    "appleleaf9": {"name": "AppleLeaf9 (Yang et al. 2022; includes Plant Pathology 2020/2021 + ATLDSD)",
                   "licence": "CC BY 4.0", "domain": "field (few orchards / cameras)"},
    "kashmir_apple": {"name": "Kashmir indigenous apple (Yatoo & Sharma 2024)", "licence": "CC BY 4.0",
                      "domain": "field (J&K orchards, 256 px)"},
}
FIELD_SOURCES = ("plantdoc", "appleleaf9", "kashmir_apple")
CATEGORY = {"diagnosable": "trained", "limited": "partial", "related": "partial", "none": "symptom_only"}


def _sci_key(name: str) -> str:
    return " ".join(name.replace("×", "").split()[:2]).lower()


def build() -> dict:
    kb = load_kb()
    idx = pd.read_csv(RESULTS / "image_index.csv")
    counts = idx.groupby(["class_key", "source", "split"]).size()
    ev_path = RESULTS / "multicrop_eval.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    s10 = json.loads(STAGE10_REGISTRY.read_text(encoding="utf-8"))["plants"] if STAGE10_REGISTRY.exists() else {}
    s10_by_sci = {_sci_key(v["scientific_name"]): (k, v) for k, v in s10.items()}

    plants = []
    for p in kb.plants:
        cov = kb.model_coverage(p["model_crop"])
        classes = cov["classes"] or cov["related_classes"]
        data = {}
        for k in classes:
            data[k] = {f"{src}/{sp}": int(counts.get((k, src, sp), 0))
                       for src in PROVENANCE for sp in ("train", "val", "test") if counts.get((k, src, sp), 0)}
        sources = sorted({key.split("/")[0] for v in data.values() for key in v})
        field_test = sum(n for v in data.values() for key, n in v.items()
                         if key.split("/")[0] in FIELD_SOURCES and key.endswith("/test"))
        crop_keys = kb.crop_keys(p["model_crop"]) if p["model_crop"] else []
        kb_dis = [d for d in kb.diseases if d["crop"] in crop_keys and d["pathogen_type"] != "none"]
        s10_hit = s10_by_sci.get(_sci_key(p["scientific_name"]))
        field_eval = None
        if ev and cov["level"] in ("diagnosable", "limited"):
            field_eval = ev.get("per_crop", {}).get(p["model_crop"], {}).get("field")
        lim = []
        if cov["level"] == "limited":
            lim.append("only " + ("healthy" if not cov["diseases"] else "disease") + " classes trained; no named results")
        if cov["level"] == "related":
            lim.append(f"no photos of this crop; uses {', '.join(cov['related_crops'])} classes as leads only")
        if cov["level"] == "none" and kb_dis:
            lim.append("symptom matching only; always 'Unknown disease' with leads")
        if cov["level"] == "none" and not kb_dis:
            lim.append("no diseases in the database")
        if cov["level"] in ("diagnosable", "limited") and field_test < 30:
            lim.append(f"field test photos: {field_test} (too few to estimate field accuracy per disease)")
        if p["model_crop"] == "apple":
            lim.append("apple field classes come mostly from one source each (e.g. 3,166/3,172 black-rot field "
                       "photos are Plant Pathology 2021, one orchard); no independent black-rot field test exists")
        lim.append("database entries are not expert-reviewed")
        category = CATEGORY[cov["level"]] if (cov["level"] != "none" or kb_dis) else "unsupported"
        plants.append({
            "common_name": p["common_name"], "scientific_name": p["scientific_name"], "family": p["family"],
            "model_crop": p["model_crop"], "support_category": category, "model_coverage": cov["level"],
            "trained_classes": cov["classes"], "related_classes": cov["related_classes"],
            "image_counts": data, "data_sources": [{"source": s, **PROVENANCE[s]} for s in sources],
            "field_test_photos": field_test, "field_evaluation": field_eval,
            "kb_diseases_stage11": [d["key"] for d in kb_dis],
            "kb_diseases_without_photos": [d["key"] for d in kb_dis if not d.get("model_class")],
            "stage10_kb": None if not s10_hit else {"plant_id": s10_hit[0],
                                                     "conditions_sourced": s10_hit[1].get("conditions_sourced", [])},
            "limitations": lim,
        })
    summary = defaultdict(int)
    for p in plants:
        summary[p["support_category"]] += 1
    return {"generated_by": "stage11/coverage.py", "evaluation_file": "multicrop_eval.json" if ev else None,
            "summary": dict(summary), "plants": plants}


def to_markdown(reg: dict) -> str:
    lines = ["# Stage 11 crop coverage registry", "",
             "Generated by `stage11/coverage.py` from the database, the image index and the multi-crop evaluation. "
             "Do not edit by hand.", "",
             "Categories: **trained** = classifier trained on healthy and diseased photos of the crop; **partial** = "
             "only some conditions, or a related crop only (leads, never a named result); **symptom_only** = "
             "database diseases matched by detected symptoms (always 'Unknown disease' with leads); "
             "**unsupported** = nothing in the database.", "",
             "Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(reg["summary"].items())), "",
             "| Plant | Scientific name | Category | Trained classes | Train / val / test photos (all sources) | "
             "Field test photos | Field top-1 (crop given) | DB diseases (no photos) | Stage 10 KB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(reg["plants"], key=lambda r: (["trained", "partial", "symptom_only", "unsupported"]
                                                  .index(r["support_category"]), r["common_name"])):
        tot = {"train": 0, "val": 0, "test": 0}
        for v in p["image_counts"].values():
            for k, n in v.items():
                tot[k.split("/")[1]] += n
        fe = p["field_evaluation"]
        fe_txt = f"{100 * fe['top1']:.0f}% (n={fe['n']})" if fe and fe.get("n") else "–"
        cls = p["trained_classes"] or ([f"({c})" for c in p["related_classes"]])
        lines.append(f"| {p['common_name']} | *{p['scientific_name']}* | {p['support_category']} | "
                     f"{len(cls)}: {', '.join(cls) if cls else '–'} | "
                     f"{tot['train']} / {tot['val']} / {tot['test']} | {p['field_test_photos']} | {fe_txt} | "
                     f"{len(p['kb_diseases_stage11'])} ({len(p['kb_diseases_without_photos'])}) | "
                     f"{'yes' if p['stage10_kb'] else '–'} |")
    lines += ["", "## Data sources", ""]
    for k, v in PROVENANCE.items():
        lines.append(f"- **{k}**: {v['name']} — licence {v['licence']}; domain {v['domain']}")
    lines += ["", "## Limitations per plant", ""]
    for p in reg["plants"]:
        lines.append(f"- **{p['common_name']}** ({p['support_category']}): " + "; ".join(p["limitations"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    reg = build()
    js, md = json.dumps(reg, indent=1, ensure_ascii=False) + "\n", to_markdown(reg)
    if a.check:
        ok = OUT_JSON.exists() and OUT_JSON.read_text(encoding="utf-8") == js and \
            OUT_MD.exists() and OUT_MD.read_text(encoding="utf-8") == md
        print("coverage registry", "up to date" if ok else "OUT OF DATE")
        return 0 if ok else 1
    OUT_JSON.write_text(js, encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(reg["summary"], "->", OUT_MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
