"""Farmer-facing summary of one assessment (scope change SC-M5-02, owner decision 2026-10-15).

Built ONLY from the validated report-engine output, the knowledge base and the visible-sign analysis; it adds no new
claims. Two shapes, matching the owner's specification:

  kind "known"     the frozen model accepted the photo as a disease class of the confirmed crop with a sourced record:
                   crop, scientific name, soil and nutrients, temperature, signs seen, the disease, what to do;
  kind "possible"  otherwise: the same crop facts and signs, up to three possible diseases, the checks that would
                   confirm each, and what to do for each;
  kind "none"      no candidate could be ranked (e.g. a healthy-looking leaf, or a plant outside the knowledge base).

"What to do" lists the sourced practices the report engine emitted (non-chemical). Spray products named by a source
are listed by name only, never with a dose, under "ask before spraying" because no Indian registration or label
check exists in this project (report-engine policy). Nothing here confirms a diagnosis.
"""
from __future__ import annotations

NUTRIENT_LABEL = {"nitrogen": "Nitrogen (N)", "phosphorus": "Phosphorus (P)", "potassium": "Potassium (K)",
                  "organic_manure": "Organic manure (farmyard manure / compost)", "zinc": "Zinc (Zn)", "boron": "Boron (B)",
                  "manganese": "Manganese (Mn)", "calcium": "Calcium (Ca)", "iron": "Iron (Fe)", "sulphur": "Sulphur (S)",
                  "magnesium": "Magnesium (Mg)"}
CAUSE = {"fungal": "Fungus", "bacterial": "Bacteria", "viral": "Virus", "oomycete": "Water mould (oomycete)", "mite": "Mite (pest)",
         "insect": "Insect (pest)", "nutrient_deficiency": "Nutrient shortage", "fungal_complex": "Several fungi"}


def short_publisher(pub: str) -> str:
    """'TNAU Agritech Portal, Tamil Nadu ...' -> 'TNAU Agritech Portal'; never cuts inside brackets."""
    t = pub.split(",")[0]
    return t.split(" (")[0] if t.count("(") > t.count(")") else t


def _src(kb, claim) -> list[str]:
    seen = []
    for e in claim["evidence"]:
        t = short_publisher(kb.sources[e["source_id"]]["publisher_or_author"])
        if t not in seen:
            seen.append(t)
    return seen


def _fact(kb, claim):
    return None if not claim else {"text": claim["text"], "sources": _src(kb, claim)}


def crop_facts(kb, pid: str | None, typed_name: str | None) -> dict:
    if not pid:
        return {"name": typed_name, "scientific_name": None, "family": None, "in_knowledge_base": False,
                "temperature": None, "soil": None, "nutrients": None}
    p = kb.plants[pid]
    cr = kb.crop_requirements.get(pid) or {}
    nut = cr.get("nutrients")
    return {"name": p["common_names"][0]["name"], "scientific_name": p["scientific_name"], "family": p["taxonomy"]["family"],
            "in_knowledge_base": True, "temperature": _fact(kb, cr.get("temperature")), "soil": _fact(kb, cr.get("soil")),
            "nutrients": None if not nut else {"needed": [NUTRIENT_LABEL[n] for n in nut["needed"]], **_fact(kb, nut["claim"])},
            "notes": cr.get("notes", [])}


def _disease(kb, rep, c) -> dict:
    cid = c["kb_disease_id"]
    cond = kb.conditions[cid]
    sym = {s["symptom_id"]: s for s in rep["visible_symptoms"]}
    matched = [sym[s]["label"]["en"] for s in c["supporting_symptoms"] if s in sym]
    typical = [f["claim"]["text"] for f in cond["features"] if f["role"] in ("distinguishing", "typical")]
    checks = []
    for did in cond["diagnostic_ids"]:
        d = kb.diagnostics[did]
        if d["curation_status"] != "sourced":
            checks.append({"name": d["name"], "how": d["description"], "shows": None, "cannot_show": None, "sourced": False})
        else:
            checks.append({"name": d["name"], "how": d["description"], "shows": d["what_it_can_establish"],
                           "cannot_show": d["what_it_cannot_establish"], "sourced": True})
    steps, sprays = [], []
    for mid in cond["management_ids"]:
        m = kb.management[mid]
        item = {"text": m["summary"], "kind": m["category"], "outside_india": not m["geographic_applicability"].startswith("India")}
        (sprays if m["category"] in ("chemical", "biological", "nutrient") else steps).append(item)
    cur = cond["curability"]
    return {"condition_id": cid, "name": c["candidate_name"]["en"], "cause": CAUSE.get(cond["cause_category"], cond["cause_category"]),
            "scientific_cause": (cond["scientific_cause"] or {}).get("name"), "rank_score": c["ranking_score"],
            "ranking_method": c["ranking_method"], "signs_matched": matched, "typical_signs": typical, "checks": checks,
            "what_to_do": steps, "sprays_named_by_source": sprays,
            "curability": {"status": cur["status"], "text": cur["claim"]["text"] if cur.get("claim") else None},
            "uncertainty": c["uncertainty_statement"]["en"], "sources": sorted({short_publisher(kb.sources[s]["publisher_or_author"])
                                                                                for s in c["source_ids"] if s in kb.sources})}


def build(kb, rep: dict, workflow: dict, analysis: dict | None) -> dict:
    pid = workflow["plant_selection"].get("plant_id")
    cands = rep["candidate_diagnoses"]
    known = rep["report_type"] == "known_disease" and cands and cands[0]["ranking_method"] == "model_class_evidence"
    healthy_cls = (rep["report_type"] == "known_disease" and not cands and rep["model_prediction"]["ran"]
                   and "healthy" in (rep["model_prediction"]["predicted_class"] or ""))
    kind = "known" if known else ("possible" if cands else ("healthy_looking" if healthy_cls else "none"))
    mp, osr = rep["model_prediction"], rep["osr_status"]
    if not mp["ran"]:
        model = {"ran": False, "accepted": None, "note": "The photo disease model was not run for this plant.",
                 "reason": workflow["routing"]["reason"]}
    elif osr["decision"] == "known":
        model = {"ran": True, "accepted": True, "note": "The photo matched one of the model's known disease classes closely enough."}
    else:
        model = {"ran": True, "accepted": False,
                 "note": ("The photo did not match the model's training photos closely enough for a firm answer (this model was trained "
                          "on laboratory leaf photos and rejects most field photos). Its closest matches are used only as possible causes.")}
    notes = ["Nothing in this report is a confirmed diagnosis. Confirm with the checks listed, or with your local KVK / agriculture officer.",
             "Spray products are never recommended here, and no doses are given: ask your local agriculture officer which product is registered and safe."]
    if kind == "healthy_looking":
        notes.insert(0, "The model matched this photo to a healthy leaf of this crop. That does not prove the plant is free of disease, "
                        "pests or nutrient problems; check other leaves and the stem, and look again in a few days.")
    if kind == "none":
        notes.insert(0, rep["insufficient_evidence_statement"]["en"] if rep.get("insufficient_evidence_statement") else
                     "No disease could be suggested from this photo and the information given.")
    return {"kind": kind, "crop": crop_facts(kb, pid, workflow["plant_selection"].get("plant_name")),
            "signs": analysis, "model": model, "diseases": [_disease(kb, rep, c) for c in cands[:3]], "notes": notes}
