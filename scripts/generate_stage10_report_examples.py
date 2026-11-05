"""Stage 10 / Milestone 3 — end-to-end report examples (read-only with respect to Stages 1-9).

  python -P scripts/generate_stage10_report_examples.py

Runs the UNCHANGED frozen predictor (via FrozenOSRAdapter) on PlantVillage VALIDATION images only (data/splits/val.csv;
the final-test guard refuses anything in data/splits/test.csv) and builds reports with the deterministic engine.
Image selection rule (documented, not tuned): the first validation image of the named class, in sorted path order, whose
frozen OSR decision matches the scenario. User answers in these examples are SYNTHETIC and are labelled as such.
The mango example uses a synthetic placeholder image generated here (no permitted mango photographs exist in the project),
and the model is not run on it.

Writes results/stage10/report_examples/<name>.report.json, <name>.trace.json and SUMMARY.md.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage10.knowledge_base.kb import load_kb  # noqa: E402
from stage10.osr.osr_adapter import FrozenOSRAdapter  # noqa: E402
from stage10.report_engine.engine import build_report  # noqa: E402
from stage10.report_engine.inputs import ImageInfo, Region, ReportRequest, image_info_from_file  # noqa: E402

OUT = ROOT / "results" / "stage10" / "report_examples"
TS = "2026-10-14T12:00:00+00:00"


def val_images(cls: str) -> list[str]:
    with open(ROOT / "data/splits/val.csv", newline="", encoding="utf-8") as f:
        return sorted(r["image_path"] for r in csv.DictReader(f) if r["class_name"] == cls and r["split"] == "val")


def pick(adapter, cls: str, decision: str, want_class: str | None = None):
    for p in val_images(cls):
        a = adapter.assess(ROOT / p)
        if a["osr_status"]["decision"] == decision and (want_class is None or a["model_prediction"]["predicted_class"] == want_class):
            return p, a
    raise SystemExit(f"no validation image of {cls} with decision {decision}")


def synthetic_png() -> tuple[ImageInfo, bytes]:
    import hashlib

    from PIL import Image
    im = Image.new("RGB", (256, 256))
    im.putdata([(40 + x // 4, 110 + y // 6, 40) for y in range(256) for x in range(256)])
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b = buf.getvalue()
    return ImageInfo(hashlib.sha256(b).hexdigest(), 256, 256, "PNG"), b


def md(name: str, desc: str, img: str, r: dict, trace: dict) -> str:
    L = [f"## {name}", "", desc, "", f"- image: `{img}`", f"- report_type: **{r['report_type']}**, report_id `{r['report_id']}`",
         f"- plant: {r['plant_identity']['common_name']['en']} (user_confirmation={r['plant_identity']['user_confirmation']}, "
         f"in_model_scope={r['plant_identity']['in_model_scope']})",
         f"- model: ran={r['model_prediction']['ran']}, class={r['model_prediction']['predicted_class']}, OSR={r['osr_status']['decision']}"
         + (f", score {r['osr_status']['score']:.3f} vs frozen {r['osr_status']['frozen_threshold']:.4f}" if r['osr_status']['score'] is not None else ""),
         f"- OSR message: {r['osr_status']['message']['en']}"]
    if r.get("insufficient_evidence_statement"):
        L.append(f"- **Insufficient evidence:** {r['insufficient_evidence_statement']['en']}")
    for c in r["candidate_diagnoses"]:
        L.append(f"- {c['candidate_id']} **{c['candidate_name']['en']}** ({c['cause_category']}; method {c['ranking_method']}, "
                 f"score {c['ranking_score']}, not a probability; crop_compatible={c['crop_compatible']}; sources {', '.join(c['source_ids'])})")
        for x in c["conflicting_or_missing"]:
            L.append(f"  - {x['en']}")
    for sec in ("recommended_tests", "treatment_guidance", "nutrition_guidance", "immediate_actions"):
        for x in r[sec]:
            L.append(f"- [{sec}] {x['recommendation_id']} ({x['category']}/{x['subcategory']}, {x['geographic_applicability']}, "
                     f"sources: {', '.join(x['source_ids']) or 'none'}): {x['text']['en']}")
    L.append(f"- withheld management (not in report advice): {[w['management_id'] for w in trace['withheld_management']]}")
    L.append(f"- evidence sources: {[s['source_id'] for s in r['evidence_sources']]}; trace entries: {len(trace['entries'])}")
    L.append("- safety warnings: " + " | ".join(w["en"] for w in r["safety_warnings"]))
    if r["language"] == "hi":
        L.append("- Hindi (pending fluent-speaker review): " + r["osr_status"]["message"].get("hi", "")[:120] + " …")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kb = load_kb()
    ad = FrozenOSRAdapter()
    cases = []

    p, a = pick(ad, "Tomato___Early_blight", "known", "Tomato___Early_blight")
    cases.append(("1_known_tomato_early_blight", "Known-class case. Validation image; user confirms Tomato (Maharashtra) and reports "
                  "(SYNTHETIC answers) ringed spots and older leaves affected first.", p,
                  ReportRequest(image=image_info_from_file(ROOT / p), plant_name="Tomato", plant_confirmation="confirmed",
                                reported_features={"FEAT-CONCENTRIC-RINGS": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True},
                                model_assessment=a, region=Region("Maharashtra"), generated_at=TS)))
    p, a = pick(ad, "Tomato___Tomato_mosaic_virus", "unknown")
    req = dict(image=image_info_from_file(ROOT / p), plant_name="Tomato", plant_confirmation="confirmed",
               reported_features={"FEAT-INTERVEINAL-CHLOROSIS-GREEN-MARGINS": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True},
               model_assessment=a, region=Region("Tamil Nadu"), generated_at=TS)
    cases.append(("2_unfamiliar_tomato_investigation", "Unfamiliar-class case. Validation-unknown image (research label: Tomato mosaic "
                  "virus, which the knowledge base does NOT cover). User answers are SYNTHETIC (interveinal yellowing, older leaves "
                  "first). Illustrates that a supported candidate can still be the wrong cause: it stays an unconfirmed hypothesis.",
                  p, ReportRequest(**req)))
    cases.append(("3_unfamiliar_tomato_hindi", "Same as example 2, Hindi report (draft translations pending fluent-speaker review).",
                  p, ReportRequest(**{**req, "language": "hi"})))
    cases.append(("4_abstention_single_symptom", "Abstention. Same unfamiliar image; user reports only one symptom (SYNTHETIC).", p,
                  ReportRequest(**{**req, "reported_features": {"FEAT-YELLOW-HALO": True}})))
    info, _ = synthetic_png()
    cases.append(("5_mango_knowledge_only", "Knowledge coverage without image capability. SYNTHETIC placeholder image (generated, not a "
                  "photograph); model not run; user confirms Mango and reports sunken black fruit spots during ripening (SYNTHETIC).",
                  "synthetic-placeholder.png",
                  ReportRequest(image=info, plant_name="Mango", plant_confirmation="confirmed",
                                reported_features={"FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": True, "FEAT-FRUIT-LESIONS": True},
                                region=Region("Karnataka"), generated_at=TS)))
    p, a = pick(ad, "Cherry_(including_sour)___healthy", "unknown")
    cases.append(("6_plant_outside_knowledge_base", "Plant outside the knowledge base. Validation-unknown cherry image; user confirms "
                  "'Cherry'.", p,
                  ReportRequest(image=image_info_from_file(ROOT / p), plant_name="Cherry", plant_confirmation="confirmed",
                                model_assessment=a, generated_at=TS)))
    cases.append(("7_no_plant_cross_crop", "Unidentified plant. Same tomato-mosaic image, no plant given; SYNTHETIC answers match "
                  "black rot of two different hosts.", cases[1][2],
                  ReportRequest(image=image_info_from_file(ROOT / cases[1][2]),
                                reported_features={"FEAT-BLACK-SPECKS-IN-LESION": True, "FEAT-FRUIT-MUMMY": True, "FEAT-FRUIT-LESIONS": True},
                                model_assessment=cases[1][3].model_assessment, generated_at=TS)))

    summary = ["# AgroVaidya report examples (Stage 10, Milestone 3)", "",
               "Generated by `scripts/generate_stage10_report_examples.py`. Validation images only; user answers are synthetic. "
               "Reports are research-prototype outputs: not expert-reviewed, not diagnoses.", ""]
    for name, desc, img, rq in cases:
        res = build_report(rq, kb)
        (OUT / f"{name}.report.json").write_text(res.to_json(), encoding="utf-8")
        (OUT / f"{name}.trace.json").write_text(json.dumps(res.evidence_trace, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        summary.append(md(name, desc, img, res.report, res.evidence_trace))
        print(f"{name}: {res.report['report_type']} | candidates {[c['candidate_name']['en'] for c in res.report['candidate_diagnoses']]} "
              f"| OSR {res.report['osr_status']['decision']} | sources {len(res.report['evidence_sources'])} | trace {len(res.evidence_trace['entries'])}")
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
