"""Engine-level checks run on every generated report (in addition to the schema validator, which is unchanged in meaning).

engine_errors(report, trace, kb, request) -> list[str]   (empty = OK)

 1. validate_report: JSON Schema + semantic rules, with the frozen threshold.
 2. Traceability: every source_id used anywhere is in evidence_sources; every evidence source and every sourced candidate,
    recommendation and taxonomy element has trace entries; every trace passage is a verbatim substring of the hash-checked
    cached source text, and is the passage the knowledge base records for that claim.
 3. Candidates: <= max, never confirmed, score not a probability, unvalidated; supporting symptoms are only real
    observations (user report / image analysis), never knowledge-base typical symptoms.
 4. Plant identity: if the user did not confirm the plant, every candidate is preliminary and no management is given.
 5. Treatment safety: no chemical/biological recommendation (none is registration-verified); no dose; no confirmed nutrient
    deficiency; nutrient advice carries a verification test; foreign guidance is labelled; consequential advice is sourced.
 6. Wording: no dose-like text, no promise of recovery; unknown OSR results use the established wording; the frozen model's
    output is copied unchanged.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from stage10.knowledge_base.kb import DOSE, ROOT, norm
from stage10.schemas.validate import ReportValidationError, _walk_localized, stage10_config, validate_report

BANNED = re.compile(r"\b(guarantee[sd]?|definitely|certainly|100 ?percent|will cure|fully cured|confirmed diagnosis of)\b", re.I)
UNFAMILIAR = "Potentially unfamiliar to this model; further investigation is recommended."
REC_SECTIONS = ("recommended_tests", "treatment_guidance", "nutrition_guidance", "soil_guidance", "immediate_actions")


@lru_cache(maxsize=64)
def _cached_text(cache_file: str) -> str:
    return norm((ROOT / cache_file).read_text(encoding="utf-8"))


def _all_source_ids(o, out=None):
    out = set() if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "source_ids":
                out.update(v)
            elif k == "source_id" and isinstance(v, str):
                out.add(v)
            else:
                _all_source_ids(v, out)
    elif isinstance(o, list):
        for v in o:
            _all_source_ids(v, out)
    return out


def _kb_passages(kb) -> set:
    from stage10.knowledge_base.kb import _evidence
    return {(e["source_id"], e["passage"]) for k in ("plants", "conditions", "diagnostics", "management") for r in kb.raw[k]
            for e in _evidence(r)}


def engine_errors(r: dict, trace: dict, kb, req) -> list[str]:
    from stage10.report_engine.engine import frozen_threshold

    err: list[str] = []
    cfg = stage10_config()
    try:
        validate_report(r, frozen_threshold=frozen_threshold())
    except ReportValidationError as e:
        err += [f"validator: {x}" for x in e.errors]
    # 2. traceability
    ev_ids = {s["source_id"] for s in r["evidence_sources"]}
    for sid in sorted(_all_source_ids({k: v for k, v in r.items() if k != "evidence_sources"}) - ev_ids):
        err.append(f"trace: {sid} used but not in evidence_sources")
    known = _kb_passages(kb)
    traced = set()
    for t in trace["entries"]:
        for e in t["evidence"]:
            sid = e["source_id"]
            traced.add(sid)
            if sid not in kb.sources:
                err.append(f"trace: unknown source {sid}")
                continue
            if (sid, e["passage"]) not in known:
                err.append(f"trace: passage for {sid} is not a knowledge-base evidence passage")
            if norm(e["passage"]) not in _cached_text(kb.sources[sid]["verification"]["cache_file"]):
                err.append(f"trace: passage not found verbatim in the cached text of {sid}")
    withheld_src = {s for w in trace["withheld_management"] for s in w["source_ids"]}
    for sid in ev_ids - traced - withheld_src:
        err.append(f"trace: evidence source {sid} has no traced claim")
    paths = {t["report_path"] for t in trace["entries"]}

    def need_trace(prefix, sids, what):
        if sids and not any(p.startswith(prefix) for p in paths):
            err.append(f"trace: {what} cites sources but has no trace entry")
        cited = {e["source_id"] for t in trace["entries"] if t["report_path"].startswith(prefix) for e in t["evidence"]}
        for s in set(sids) - cited:
            err.append(f"trace: {what} cites {s} without a traced passage")
    for c in r["candidate_diagnoses"]:
        need_trace(f"candidate_diagnoses/{c['candidate_id']}", c["source_ids"], c["candidate_id"])
    for sec in REC_SECTIONS:
        for x in r[sec]:
            need_trace(f"{sec}/{x['recommendation_id']}", x["source_ids"], x["recommendation_id"])
            if x["consequential"] and not x["source_ids"]:
                err.append(f"{x['recommendation_id']}: consequential recommendation without a source")
    tax = {s for v in r["plant_identity"]["taxonomy"].values() for s in v["source_ids"]}
    need_trace("plant_identity/taxonomy", tax, "taxonomy")
    # 3. candidates
    cands = r["candidate_diagnoses"]
    if len(cands) > cfg["candidates"]["max_candidates"]:
        err.append("candidates: more than max_candidates")
    sym = {s["symptom_id"]: s for s in r["visible_symptoms"]}
    for c in cands:
        if c["status"] != "candidate" or c["ranking_score_is_probability"] is not False or c["validation_status"] != "unvalidated":
            err.append(f"{c['candidate_id']}: must be an unvalidated candidate with a non-probability score")
        if "confirmed" in c["candidate_name"]["en"].lower():
            err.append(f"{c['candidate_id']}: candidate name implies confirmation")
        if c["ranking_method"] == "model_class_evidence" and c["ranking_score"] is not None:
            err.append(f"{c['candidate_id']}: model-class candidate must not carry a ranking score")
        for s in c["supporting_symptoms"]:
            if sym[s]["origin"] not in ("user_report", "image_analysis") or sym[s]["evidence_status"] not in ("observed", "inferred"):
                err.append(f"{c['candidate_id']}: supporting symptom {s} is not an observation")
        if c["kb_disease_id"] not in kb.conditions or kb.conditions[c["kb_disease_id"]]["curation_status"] != "sourced":
            err.append(f"{c['candidate_id']}: candidate must be a sourced knowledge-base condition")
    # 4. plant identity
    if r["plant_identity"]["user_confirmation"] != "confirmed":
        for c in cands:
            if c["crop_compatible"] or "crop_mismatch_note" not in c:
                err.append(f"{c['candidate_id']}: plant unconfirmed, candidate must be marked preliminary")
        if any(r[s] for s in ("treatment_guidance", "nutrition_guidance", "soil_guidance")):
            err.append("management guidance given although the plant is unconfirmed")
    if r["plant_identity"]["species_level_claimed"]:
        err.append("species-level identification must not be claimed")
    # 5. treatment safety
    for sec in REC_SECTIONS:
        for x in r[sec]:
            rid = x["recommendation_id"]
            if x["subcategory"] in ("chemical", "biological", "fertilizer"):
                err.append(f"{rid}: {x['subcategory']} recommendation emitted (no registration/label verification exists)")
            if x["dose"] is not None:
                err.append(f"{rid}: dose emitted")
            if x["category"] == "confirmed_nutrient_deficiency":
                err.append(f"{rid}: confirmed nutrient deficiency emitted without recorded test evidence")
            if x["category"] == "suspected_nutrient_issue" and ("verification_test" not in x or "Suspected only" not in x["text"]["en"]):
                err.append(f"{rid}: nutrient advice must stay a hypothesis with a verification test")
            if x["geographic_applicability"] == "international_not_verified_for_India" and not any(
                    "not verified for India" in n["en"] for n in x.get("risk_notes", [])):
                err.append(f"{rid}: foreign guidance lacks a 'not verified for India' label")
    # 6. wording + preserved model output
    for path, t in _walk_localized({k: v for k, v in r.items() if k not in ("evidence_sources", "model_prediction")}):
        for lang, s in t.items():
            if DOSE.search(s):
                err.append(f"{path}.{lang}: dose-like text")
            if BANNED.search(s):
                err.append(f"{path}.{lang}: overclaiming wording")
    if r["osr_status"]["decision"] == "unknown" and not r["osr_status"]["message"]["en"].startswith(UNFAMILIAR):
        err.append("osr_status: unknown result must use the established 'potentially unfamiliar' wording")
    if req.model_assessment is not None:
        if r["model_prediction"] != req.model_assessment["model_prediction"]:
            err.append("model_prediction differs from the frozen adapter output")
        a = req.model_assessment["osr_status"]
        for k in a:
            if k in ("message", "interpretation_note"):
                if r["osr_status"][k]["en"] != a[k]["en"]:
                    err.append(f"osr_status.{k} English text changed")
            elif r["osr_status"][k] != a[k]:
                err.append(f"osr_status.{k} differs from the frozen adapter output")
    # 7. language-independent content keys (D-6) and withheld management text
    from stage10.report_engine.engine import messages
    from stage10.report_engine.localize import _resolve
    cat = messages()["messages"]
    keyed_paths = set()
    for ck in r.get("content_keys", []):
        try:
            obj = _resolve(r, ck["path"])
        except (KeyError, IndexError, ValueError, TypeError):
            err.append(f"content_keys: path {ck['path']} does not resolve")
            continue
        keyed_paths.add(ck["path"])
        m = cat.get(ck["key"])
        if m is None or not isinstance(obj, dict):
            err.append(f"content_keys: unknown key {ck['key']} at {ck['path']}")
            continue
        for lang, txt in obj.items():
            if txt != m[lang].format(**ck["params"]):
                err.append(f"content_keys: {ck['path']}.{lang} differs from catalogue key {ck['key']}")
    if r["language"] != "en":
        for path, t in _walk_localized({k: v for k, v in r.items() if k not in ("content_keys", "review_status")}):
            if r["language"] in t and path not in keyed_paths:
                err.append(f"{path}: translated text without a content key")
    dump = json.dumps({k: v for k, v in r.items() if k != "evidence_sources"}, ensure_ascii=False)
    for w in trace["withheld_management"]:
        m = kb.management[w["management_id"]]
        if m["summary"] in dump or m["claim"]["text"] in dump:
            err.append(f"withheld management {w['management_id']} text appears in the report")
    if r["report_type"] == "known_disease" and r["osr_status"]["decision"] != "known":
        err.append("known_disease report without a known OSR decision")
    if r["language"] != "en" and r["review_status"]["translation_review"].get(r["language"]) != "pending_fluent_speaker_review":
        err.append("non-English report must be marked pending fluent-speaker review")
    return err
