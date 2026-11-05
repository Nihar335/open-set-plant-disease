"""Coverage registry: what the system can and cannot do, per plant and per condition, kept separate per capability.

  python -m stage10.knowledge_base.coverage --write   # regenerate stage10/knowledge_base/coverage_registry.json
  python -m stage10.knowledge_base.coverage --check   # fail if the committed registry is stale

Knowledge-base coverage is NEVER equated with image-diagnosis capability. Image-diagnosis status comes only from the
frozen research model's scope (config.yaml, read-only) and its frozen Stage-8 results (results/final/final_numbers.json).
New plants/conditions are picked up automatically from the KB; nothing here touches the frozen predictor.
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from stage10.knowledge_base.kb import KB_DIR, ROOT, KnowledgeBase, load_kb

OUT = KB_DIR / "coverage_registry.json"
IMG_KNOWN = ("frozen_research_model_known_class: evaluated only on PlantVillage lab-style test images in the research project "
             "(Stage 3 known-test accuracy 99.24%); NOT validated on field photographs (PlantDoc known-disease accuracy about 32%) "
             "and NOT validated as an agricultural diagnostic")
IMG_HELDOUT = ("held_out_unknown_in_research: excluded from model training; the frozen model cannot name it and can only flag an "
               "image as potentially unfamiliar (OSR). Not a validated diagnosis")
IMG_NONE = "not_covered_by_frozen_model: no image-based diagnosis capability"
CAPABILITIES = {
    "knowledge_base_text": "implemented (seed v{v}); claims source-verified, not expert-reviewed",
    "kb_text_retrieval": "implemented; ranking UNVALIDATED (no independent evaluation data yet)",
    "frozen_model_osr_decision": "reused unchanged from Stage 9 (contrastive Energy, frozen threshold); research-evaluated only",
    "image_symptom_analysis": "not implemented (roadmap task M5-A5)",
    "reference_image_retrieval": "not available: no licence-verified reference images acquired",
    "report_generation": "structured deterministic report engine implemented (Milestone 3: stage10/report_engine); outputs not expert-reviewed",
    "ui_pdf": "not implemented (planned later milestones)",
    "expert_review": "none performed",
    "hindi_content": "not present in the knowledge base yet",
}


def build(kb: KnowledgeBase) -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["osr_protocol"]
    known, held = set(cfg["known_classes"]), set(cfg["unknown_near"])
    nums = json.loads((ROOT / "results/final/final_numbers.json").read_text(encoding="utf-8"))
    per = {r["unknown_class"]: r for r in nums["per_disease"]}
    cls_by_cond = {m["condition_id"]: m["class_name"] for m in kb.raw["model_classes"] if m["condition_id"]}
    conds = {}
    for cid, c in sorted(kb.conditions.items()):
        sids = sorted({e["source_id"] for e in _evidence(c)} | {e["source_id"] for m in c["management_ids"] for e in _evidence(kb.management[m])})
        cls = cls_by_cond.get(cid)
        if cls in known:
            img = {"status": "frozen_model_known_class", "class_name": cls, "note": IMG_KNOWN}
        elif cls in held:
            r = per[cls]
            img = {"status": "held_out_unknown_in_research", "class_name": cls, "note": IMG_HELDOUT,
                   "frozen_stage8_osr_reference": {"detector": "Contrastive Energy (Stage 7 head)", "auroc_vs_known": round(r["Contrastive Energy*::auroc"], 4),
                                                   "detection_at_frozen_threshold": round(r["Contrastive Energy*::detection"], 4),
                                                   "source_file": "results/final/final_numbers.json"}}
        else:
            img = {"status": "not_covered_by_frozen_model", "note": IMG_NONE}
        conds[cid] = {
            "names": c["common_names"], "condition_type": c["condition_type"],
            "plants": sorted({a["plant_id"] for a in c["affected_plants"]} | set(c.get("unsourced_host_plants", []))),
            "knowledge_base_information": c["curation_status"],
            "n_claims": sum(1 for _ in _claims(c)), "sources_verified": sids,
            "management_records": len(c["management_ids"]), "diagnostic_records": len(c["diagnostic_ids"]),
            "kb_text_retrieval": "eligible (unvalidated)" if c["curation_status"] == "sourced" else "excluded (stub, unsourced)",
            "image_diagnosis": img,
            "geographic_applicability": c["geographic_applicability"],
            "review_status": c["review_status"], "expert_reviewed": c["review_status"] == "expert_reviewed",
            "known_limitations": c["known_uncertainties"], "last_verified": c["last_verified"],
        }
    plants = {}
    for pid, p in sorted(kb.plants.items()):
        pc = [cid for cid, x in conds.items() if pid in x["plants"]]
        plants[pid] = {
            "common_names": [x["name"] for x in p["common_names"]], "scientific_name": p["scientific_name"],
            "taxonomy_source_ids": sorted({e["source_id"] for e in p["taxonomy"]["evidence"]}),
            "crop_categories": p["crop_categories"],
            "conditions_sourced": [c for c in pc if conds[c]["knowledge_base_information"] == "sourced"],
            "conditions_stub_unsourced": [c for c in pc if conds[c]["knowledge_base_information"] != "sourced"],
            "sources_verified": sorted({s for c in pc for s in conds[c]["sources_verified"]} | {e["source_id"] for e in p["taxonomy"]["evidence"]}),
            "frozen_model_scope": p["frozen_model_scope"],
            "image_diagnosis": ({"status": "frozen_model_crop", "note": IMG_KNOWN} if p["frozen_model_scope"]
                                else {"status": "not_covered_by_frozen_model", "note": IMG_NONE}),
            "geographic_applicability": sorted({g for c in pc for g in conds[c]["geographic_applicability"]}),
            "review_status": p["review_status"], "expert_reviewed": False,
            "known_limitations": ["Knowledge-base coverage does not imply image-diagnosis capability.",
                                  "Seed knowledge base: condition coverage for this plant is partial."],
        }
    return {"kb_version": kb.version, "generated_from": "stage10/knowledge_base/data/*.yaml",
            "capabilities": {k: v.format(v=kb.version) for k, v in CAPABILITIES.items()},
            "summary": {"plants": len(plants), "conditions_total": len(conds),
                        "conditions_sourced": sum(1 for c in conds.values() if c["knowledge_base_information"] == "sourced"),
                        "conditions_stub": sum(1 for c in conds.values() if c["knowledge_base_information"] != "sourced"),
                        "sources_verified": len(kb.sources), "expert_reviewed_records": 0},
            "plants": plants, "conditions": conds}


def _evidence(o):
    from stage10.knowledge_base.kb import _evidence as ev
    return ev(o)


def _claims(o):
    from stage10.knowledge_base.kb import _claims as cl
    return cl(o)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    reg = build(load_kb())
    text = json.dumps(reg, indent=2, ensure_ascii=False) + "\n"
    if a.write:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}: {reg['summary']}")
        return
    ok = OUT.exists() and OUT.read_text(encoding="utf-8") == text
    print("COVERAGE REGISTRY:", "PASS (up to date)" if ok else "FAIL (stale or missing; run --write)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
