"""Text/feature-based candidate retrieval over the verified knowledge base (UNVALIDATED ranking).

Input: the plant (name/synonym/scientific name, optional), observed features (feature_id -> True present / False absent),
guided-question answers (question_id -> option_id), optional region. Model outputs are NOT used here.

Ranking (a-priori weights from stage10/config.yaml, not tuned on any data):
  support  = sum of weights of observed-present features that a sourced condition record describes
             (distinguishing 2, typical 1, occasional 0.5, environment 0.5)
  conflict = sum of penalties for features the user reported ABSENT that the record calls distinguishing/typical
  score    = support - conflict        (ordinal ranking score; NOT a probability)
Eligibility: score > min_score_exclusive; sourced records only; when the plant is known, only conditions sourced for that plant; at least
`min_supporting_symptom_features` observed symptom features (environment features do not count).
Returns <= max_candidates, never padded; `insufficient_evidence` when nothing qualifies.
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from stage10.knowledge_base.kb import ROOT, KnowledgeBase, load_kb

RANKING_NOTE = ("Ranking score is an ordinal knowledge-base match score, not a probability. Retrieval is UNVALIDATED: it has "
                "not been evaluated on independent data.")
INSUFFICIENT = ("Insufficient evidence to rank a reliable disease candidate. Additional observations or laboratory testing are "
                "recommended.")


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "stage10" / "config.yaml").read_text(encoding="utf-8"))["candidates"]


@dataclass
class Query:
    plant: str | None = None
    observed: dict | None = None          # feature_id -> bool
    answers: dict | None = None           # question_id -> option_id
    region: str | None = None             # free text state/district (used only to annotate applicability)


def observations_from(kb: KnowledgeBase, q: Query) -> dict:
    obs = dict(q.observed or {})
    for qid, opt in (q.answers or {}).items():
        if qid not in kb.questions:
            raise ValueError(f"unknown question {qid}")
        o = next((x for x in kb.questions[qid]["options"] if x["option_id"] == opt), None)
        if o is None:
            raise ValueError(f"unknown option {opt} for {qid}")
        for f, v in o["sets"].items():
            obs.setdefault(f, v)              # explicit observations win over questionnaire answers
    for f in obs:
        if f not in kb.features:
            raise ValueError(f"unknown feature {f}")
    return obs


def _ev_ids(claim) -> list[str]:
    return sorted({e["source_id"] for e in claim["evidence"]})


def retrieve(q: Query, kb: KnowledgeBase | None = None) -> dict:
    kb = kb or load_kb()
    cfg = _cfg()
    rc = cfg["retrieval"]
    obs = observations_from(kb, q)
    plant_id = kb.resolve_plant(q.plant)
    status_notes = []
    if q.plant and plant_id is None:
        status_notes.append(f"Plant '{q.plant}' is not in the knowledge base; no crop-specific candidates can be ranked.")
        return {"status": "plant_not_in_knowledge_base", "plant_id": None, "candidates": [], "insufficient_evidence_statement": INSUFFICIENT,
                "questions": [], "notes": status_notes, "ranking_note": RANKING_NOTE, "kb_version": kb.version, "validation_status": "unvalidated"}
    if plant_id is None:
        status_notes.append("Plant not confirmed: candidates may span crops; confirm the crop before acting on any suggestion.")
    env = {f for f, x in kb.features.items() if x["category"] == "environment"}
    present = {f for f, v in obs.items() if v}
    absent = {f for f, v in obs.items() if v is False}
    scored = []
    for c in kb.sourced_conditions():
        hosts = {a["plant_id"] for a in c["affected_plants"]}
        if rc["require_plant_compatibility"] and plant_id and plant_id not in hosts:
            continue
        links = {f["feature_id"]: (f, False) for f in c["features"]}
        links.update({f["feature_id"]: (f, True) for f in c["favourable_conditions"]})
        support, conflict, sup, con = 0.0, 0.0, [], []
        for fid, (link, is_env) in links.items():
            if fid in present:
                support += rc["weights"]["environment" if is_env else link["role"]]
                sup.append({"feature_id": fid, "role": "environment" if is_env else link["role"], "claim": link["claim"]["text"],
                            "source_ids": _ev_ids(link["claim"])})
            elif fid in absent and not is_env and rc["conflict_penalty"][link["role"]] > 0:
                conflict += rc["conflict_penalty"][link["role"]]
                con.append({"feature_id": fid, "role": link["role"], "reported": "absent", "claim": link["claim"]["text"],
                            "source_ids": _ev_ids(link["claim"])})
        n_symptom = sum(1 for s in sup if s["feature_id"] not in env)
        if n_symptom < rc["min_supporting_symptom_features"]:
            continue
        if support - conflict <= rc["min_score_exclusive"]:
            continue                          # contradicted more than supported: not a candidate
        missing = [f["feature_id"] for f in c["features"] if f["role"] in ("distinguishing", "typical") and f["feature_id"] not in obs]
        scored.append((support - conflict, c, sup, con, missing, hosts))
    scored.sort(key=lambda x: (-x[0], x[1]["condition_id"]))
    top = scored[: cfg["max_candidates"]]
    cands = []
    for score, c, sup, con, missing, hosts in top:
        diags = [kb.diagnostics[d] for d in c["diagnostic_ids"]]
        cands.append({
            "condition_id": c["condition_id"], "names": c["common_names"], "condition_type": c["condition_type"],
            "cause_category": c["cause_category"],
            "scientific_cause": None if c["scientific_cause"] is None else {"name": c["scientific_cause"]["name"],
                                                                           "source_ids": _ev_ids(c["scientific_cause"]["claim"])},
            "affected_plants": sorted(hosts), "crop_compatible": (plant_id in hosts) if plant_id else None,
            "supporting_evidence": sup, "conflicting_evidence": con,
            "missing_information": [{"feature_id": f, "label": kb.features[f]["label"]} for f in missing],
            "suggested_diagnostics": [{"diagnostic_id": d["diagnostic_id"], "name": d["name"], "status": d["curation_status"],
                                       "can_establish": d["what_it_can_establish"], "cannot_establish": d["what_it_cannot_establish"]}
                                      for d in diags],
            "look_alikes": [la["condition_id"] for la in c["look_alikes"]],
            "geographic_applicability": c["geographic_applicability"],
            "region_note": (None if not q.region else
                            ("Evidence includes an India/India-region source." if any(g.startswith("India") for g in c["geographic_applicability"])
                             else "Evidence is from non-Indian sources only; applicability to this region is not verified.")),
            "known_uncertainties": c["known_uncertainties"], "review_status": c["review_status"],
            "ranking_method": rc["method"], "ranking_score": round(score, 3), "ranking_score_is_probability": False,
            "validation_status": "unvalidated", "source_ids": sorted({s for x in sup + con for s in x["source_ids"]}),
        })
    questions = suggest_questions(kb, cands, obs, rc["max_questions"]) if cands else []
    out = {"status": "candidates" if cands else "insufficient_evidence", "plant_id": plant_id, "candidates": cands,
           "questions": questions, "notes": status_notes, "ranking_note": RANKING_NOTE, "kb_version": kb.version,
           "validation_status": "unvalidated"}
    if not cands:
        out["insufficient_evidence_statement"] = INSUFFICIENT
    return out


def suggest_questions(kb: KnowledgeBase, cands: list, obs: dict, k: int) -> list:
    """Pick unanswered questions whose features separate the current candidates (feature described for some but not all)."""
    if len(cands) < 1:
        return []
    feat_sets = []
    for c in cands:
        rec = kb.conditions[c["condition_id"]]
        feat_sets.append({f["feature_id"] for f in rec["features"] + rec["favourable_conditions"]})
    scored = []
    for qid, qrec in kb.questions.items():
        qf = {f for o in qrec["options"] for f in o["sets"]}
        if qf and qf <= set(obs):
            continue                                          # already answered
        if len(cands) == 1:
            gain = sum(1 for f in qf if f in feat_sets[0] and f not in obs)          # confirm/refute the single candidate
        else:
            gain = sum(1 for f in qf if 0 < sum(f in s for s in feat_sets) < len(feat_sets) and f not in obs)
        if gain:
            scored.append((-gain, qid))
    return [{"question_id": qid, "text": kb.questions[qid]["text"],
             "options": [{"option_id": o["option_id"], "label": o["label"]} for o in kb.questions[qid]["options"]]}
            for _, qid in sorted(scored)[:k]]
