"""AgroVaidya deterministic report engine (Stage 10, Milestone 3).

build_report(request) -> ReportResult(report, evidence_trace)

* Inputs are structured (report_engine.inputs); nothing free-form becomes a claim.
* Every substantive diagnostic/management statement is copied from a source-verified knowledge-base claim, and the
  claim's verbatim evidence passages are recorded in `evidence_trace` against the report element that uses it.
* The frozen model's output (FrozenOSRAdapter.assess) is preserved unchanged; the engine only checks its consistency
  with the frozen rule and threshold.
* Output is a plant_health_report (stage10/schemas) and passes validate_report plus the engine's own checks
  (report_engine.checks). Identical request + configuration + knowledge base => byte-identical JSON.

Report type (schema-defined):
  insufficient_image     image failed the basic quality gate
  known_disease          OSR decision "known" and the plant is in the model's scope (user-confirmed crop, or the
                         model's own crop when the user gave none; then everything is marked preliminary)
  unknown_investigation  OSR "unknown"/"not_run" for an in-scope or unidentified plant
  out_of_model_scope     the user confirmed a plant outside the frozen model's six crops (in or outside the KB)
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from stage10.knowledge_base.coverage import build as build_coverage
from stage10.knowledge_base.kb import ROOT, KnowledgeBase, load_kb
from stage10.osr.osr_adapter import MSG_KNOWN, MSG_UNKNOWN, NOTE, RULE, crop_of_class
from stage10.report_engine import policy
from stage10.report_engine.inputs import ReportRequest, validate_request
from stage10.retrieval.kb_retrieval import Query, observations_from, retrieve, suggest_questions
from stage10.schemas.validate import stage10_config

ENGINE_NAME = "agrovaidya-report-engine"
ENGINE_VERSION = "0.3.0"
HERE = Path(__file__).resolve().parent

SYMPTOM_TYPE = {
    "FEAT-CONCENTRIC-RINGS": "concentric_rings", "FEAT-YELLOW-HALO": "halo", "FEAT-GREY-TAN-CENTRE-DARK-MARGIN": "spots",
    "FEAT-PURPLE-MARGIN-FROGEYE": "spots", "FEAT-ELONGATED-CIGAR-LESIONS": "lesion_elongated",
    "FEAT-VEIN-LIMITED-RECTANGULAR": "lesion_elongated", "FEAT-SPINDLE-GREY-CENTRE-BROWN-MARGIN": "lesion_elongated",
    "FEAT-INTERVEINAL-STRIPING": "interveinal_discoloration", "FEAT-INTERVEINAL-CHLOROSIS-GREEN-MARGINS": "interveinal_discoloration",
    "FEAT-MARGIN-SCORCH": "necrosis", "FEAT-WATER-SOAKED-LESIONS": "lesion_irregular", "FEAT-WHITE-GROWTH-UNDERSIDE": "mold_like_growth",
    "FEAT-YELLOW-TRANSLUCENT-UPPER-SPOTS": "chlorosis", "FEAT-POWDERY-BROWN-PUSTULES": "pustules",
    "FEAT-DUSTY-SPORES-UNDERSIDE": "mold_like_growth", "FEAT-RAPID-BLIGHT-DEFOLIATION": "necrosis", "FEAT-STEM-LESIONS": "lesion_irregular",
    "FEAT-FRUIT-LESIONS": "spots", "FEAT-BERRY-SPOTS-PURPLE-RING": "spots", "FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": "spots",
    "FEAT-NODE-OR-NECK-BLACK-LESIONS": "lesion_irregular", "FEAT-FRUIT-SPOTS-WHITE-HALO": "spots", "FEAT-WILTING": "wilting",
}
PLANT_PART = {"FEAT-STEM-LESIONS": "stem", "FEAT-DARK-SPOTS-WOOD-SECTION": "stem", "FEAT-NODE-OR-NECK-BLACK-LESIONS": "stem",
              "FEAT-FRUIT-LESIONS": "fruit", "FEAT-FRUIT-MUMMY": "fruit", "FEAT-BERRY-SPOTS-PURPLE-RING": "fruit",
              "FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": "fruit", "FEAT-FRUIT-SPOTS-WHITE-HALO": "fruit",
              "FEAT-WHITE-GROWTH-UNDERSIDE": "leaf_lower", "FEAT-DUSTY-SPORES-UNDERSIDE": "leaf_lower",
              "FEAT-YELLOW-TRANSLUCENT-UPPER-SPOTS": "leaf_upper", "FEAT-OLDER-LOWER-LEAVES-FIRST": "whole_plant",
              "FEAT-SINGLE-SHOOT-OR-CANE": "whole_plant", "FEAT-WILTING": "whole_plant"}
CAUSE = {"fungal_complex": "fungal", "mite": "pest", "insect": "pest"}
SOURCE_GEO = {"India_specific": "India", "India_region": "India_region", "international_not_verified_for_India": "international",
              "international_verified_for_India": "international", "global": "global"}


class ReportEngineError(RuntimeError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:8]) + (f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""))
        self.errors = errors


@dataclass
class ReportResult:
    report: dict
    evidence_trace: dict

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, sort_keys=True, indent=1)


@lru_cache(maxsize=1)
def messages() -> dict:
    return yaml.safe_load((HERE / "messages.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def frozen_threshold() -> float:
    return float(json.loads((ROOT / "results/metrics/contrastive_thresholds.json").read_text())["thresholds"]
                 ["contrastive::energy_linear_head"])


@lru_cache(maxsize=1)
def frozen_model_versions() -> dict:
    """Model identity without loading the model (same values FrozenOSRAdapter reports)."""
    root_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    man = json.loads((ROOT / "stage10/ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    return {"stage3_checkpoint_sha256": root_cfg["osr_msp"]["checkpoint_sha256"],
            "contrastive_checkpoint_sha256": json.loads((ROOT / "results/metrics/contrastive_environment.json").read_text())["checkpoint_sha256"],
            "linear_head_sha256": man["protected_stage9_files"]["models/contrastive/linear_head.npz"]}


def _pct(t: str) -> str:
    """Research figures quoted from the coverage registry: write '%' as 'percent' so no text can be read as a dose."""
    return re.sub(r"\s*%", " percent", t)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def configuration_fingerprint() -> str:
    parts = [(ROOT / "stage10/config.yaml").read_bytes(), (HERE / "messages.yaml").read_bytes(),
             (HERE / "registration_verifications.yaml").read_bytes(), ENGINE_VERSION.encode()]
    return _sha(b"\x00".join(parts))


def check_model_assessment(a: dict) -> list[str]:
    """The frozen output is preserved, never altered: only its consistency with the frozen rule is checked."""
    err = []
    mp, osr = a["model_prediction"], a["osr_status"]
    if osr["decision"] not in ("known", "unknown"):
        err.append("model_assessment: decision must be known/unknown (use model_assessment=None when not run)")
        return err
    if osr["frozen_threshold"] != frozen_threshold():
        err.append("model_assessment: frozen_threshold differs from the frozen Stage-7 threshold")
    if osr["rule"] != RULE or osr["score_name"] != "contrastive_energy":
        err.append("model_assessment: rule/score_name differ from the frozen adapter")
    if (osr["score"] <= osr["frozen_threshold"]) != (osr["decision"] == "known"):
        err.append("model_assessment: decision inconsistent with the frozen rule")
    if mp["predicted_crop"] != crop_of_class(mp["predicted_class"]):
        err.append("model_assessment: predicted_crop inconsistent with predicted_class")
    if mp["model_versions"] != frozen_model_versions():
        err.append("model_assessment: model_versions differ from the frozen model identity")
    if mp.get("softmax_is_calibrated") is not False:
        err.append("model_assessment: softmax must be marked uncalibrated")
    return err


class _Builder:
    def __init__(self, req: ReportRequest, kb: KnowledgeBase, coverage: dict):
        self.req, self.kb, self.cov = req, kb, coverage
        self.cfg = stage10_config()
        self.msg = messages()["messages"]
        self.lang = req.language
        self.trace: list[dict] = []
        self.withheld: list[dict] = []
        self.statements: list[dict] = []
        self.rec_n = 0
        self.recs = {k: [] for k in ("recommended_tests", "treatment_guidance", "nutrition_guidance", "soil_guidance", "immediate_actions")}
        self.diag_rec: dict[str, str] = {}
        self.mgmt_rec: dict[str, str] = {}
        self.warn_keys: list[str] = ["warn_not_confirmed", "warn_no_doses", "warn_no_chemicals"]
        self.keyed: dict[int, tuple] = {}
        self.nutrient_rec: dict[str, str] = {}
        self.hyb_model: dict[str, float] = {}

    # ---- text helpers -------------------------------------------------------------------------------------------
    def T(self, key: str, **params) -> dict:
        """Catalogue text; the object is remembered so its key/params are written to report.content_keys (D-6)."""
        m = self.msg[key]
        obj = {"en": m["en"].format(**params), **({"hi": m["hi"].format(**params)} if self.lang == "hi" else {})}
        self.keyed[id(obj)] = (key, dict(sorted(params.items())))
        return obj

    @staticmethod
    def L(en: str) -> dict:
        return {"en": en}

    def warn(self, key):
        if key not in self.warn_keys:
            self.warn_keys.append(key)

    def state(self, en: str | dict):
        self.statements.append(en if isinstance(en, dict) else self.L(en))
        return f"uncertainty/statements/{len(self.statements) - 1}"

    def cite(self, path: str, record_id: str, claim: dict, role: str = "supports"):
        self.trace.append({"report_path": path, "kb_record": record_id, "role": role, "claim": claim["text"],
                           "evidence": [{"source_id": e["source_id"], "passage": e["passage"]} for e in claim["evidence"]]})
        return sorted({e["source_id"] for e in claim["evidence"]})

    def next_rec(self) -> str:
        self.rec_n += 1
        return f"REC-{self.rec_n}"

    def pname(self, pid: str) -> str:
        p = self.kb.plants[pid]
        return next((n["name"] for n in p["common_names"] if n["lang"] == "en"), p["common_names"][0]["name"])

    def cname(self, cid: str) -> str:
        return " / ".join(self.kb.conditions[cid]["common_names"])

    # ---- plant identity -----------------------------------------------------------------------------------------
    def resolve_plant(self):
        r = self.req
        self.confirmed = r.plant_confirmation == "confirmed"
        self.pid = self.kb.resolve_plant(r.plant_name) if self.confirmed else None
        self.hint_pid = self.kb.resolve_plant(r.plant_name) if (not self.confirmed and r.plant_name) else None
        a = r.model_assessment
        self.osr_decision = a["osr_status"]["decision"] if a else "not_run"
        self.model_class = a["model_prediction"]["predicted_class"] if a else None
        self.model_crop = a["model_prediction"]["predicted_crop"] if a else None
        if self.confirmed:
            self.in_scope = bool(self.pid and self.kb.plants[self.pid]["frozen_model_scope"])
        else:
            self.in_scope = self.osr_decision == "known"     # model-supported scope only; flagged as unconfirmed
        self.crop_label = self.kb.plants[self.pid]["frozen_model_crop_label"] if self.pid else None

    def plant_identity(self) -> dict:
        r, kb = self.req, self.kb
        unk = {"value": None, "status": "not_assessable", "source_ids": []}
        tax = {k: dict(unk) for k in ("kingdom", "family", "genus", "species")}
        stmts, alts = [], []
        if self.pid:
            p = kb.plants[self.pid]
            ids = self.cite("plant_identity/taxonomy", self.pid,
                            {"text": f"{p['scientific_name']}: kingdom {p['taxonomy']['kingdom']}, family {p['taxonomy']['family']}, "
                                     f"genus {p['taxonomy']['genus']} (status {p['taxonomy']['status']})",
                             "evidence": p["taxonomy"]["evidence"]})
            for k in tax:
                tax[k] = {"value": p["taxonomy"][k], "status": "inferred", "source_ids": ids}
            common = self.L(self.pname(self.pid))
            level = "crop_common_name"
            stmts.append(self.L("Identity is the crop name you confirmed. The photographed plant has not been identified to species, "
                                "variety or cultivar; the taxonomy (GBIF Backbone) describes the named crop, not this specimen."))
        elif self.confirmed:
            common = self.L(r.plant_name)
            level = "crop_common_name"
            stmts.append(self.L(f"'{r.plant_name}' is not in the AgroVaidya knowledge base, so no taxonomy is given for it."))
        else:
            common = self.T("plant_unidentified")
            level = "not_assessable"
            stmts.append(self.L("You did not confirm the plant. The plant is treated as unidentified."))
            if r.plant_name:
                alts.append(self.L(f"You suggested '{r.plant_name}' but were not sure."))
        if self.model_crop and self.osr_decision == "known":
            alts.append(self.L(f"The image model's class belongs to {self.model_crop}. This is supporting evidence only, not an identification."))
        if self.confirmed and self.in_scope and self.model_crop:
            agrees = self.model_crop == self.crop_label
        else:
            agrees = None
        if not self.in_scope:
            agrees = None
        if self.req.model_assessment is None:
            note = "The image model was not run."
        elif agrees is True:
            note = f"The model's class ({self.model_class}) belongs to the crop you confirmed."
        elif agrees is False:
            note = (f"The model's class ({self.model_class}) belongs to {self.model_crop}, not to the crop you confirmed. "
                    "Your confirmation takes priority; the model's class is not applicable.")
        elif self.confirmed and not self.in_scope:
            note = "The image model only covers Apple, Corn, Grape, Pepper bell, Potato and Tomato, so its crop output is not applicable."
        else:
            note = "No user-confirmed crop to compare with; the model's crop is supporting evidence only."
        level_u = "moderate" if self.pid else "not_assessable"
        return {"user_confirmation": r.plant_confirmation, "crop_confirmed_by_user": r.plant_name if self.confirmed else None,
                "model_supporting_evidence": {"model_crop": self.model_crop, "model_class": self.model_class,
                                              "agrees_with_user": agrees, "note": self.L(note)},
                "in_model_scope": self.in_scope, "identification_level": level, "common_name": common, "taxonomy": tax,
                "species_level_claimed": False, "variety_or_cultivar": None, "growth_stage": None, "alternatives": alts,
                "uncertainty": {"level": level_u, "statements": stmts}}

    # ---- symptoms -------------------------------------------------------------------------------------------------
    def symptoms(self):
        kb, r = self.kb, self.req
        self.sym, self.sym_of = [], {}
        n = 0
        for ob in r.image_observations:
            n += 1
            sid = f"SYM-{n}"
            self.sym.append({
                "symptom_id": sid, "symptom_type": SYMPTOM_TYPE.get(ob.feature_id, "other"),
                "label": self.L(kb.features[ob.feature_id]["label"]), "evidence_status": "inferred", "origin": "image_analysis",
                "input_channel": "image_analysis",
                "location": {"plant_part": ob.plant_part, "bbox": list(ob.bbox) if ob.bbox else None},
                "affected_area_fraction": ob.affected_area_fraction, "measurement_method": ob.measurement_method,
                "visual_evidence": self.L(f"Located by automated analyser '{ob.analyser}' (unvalidated)."),
                "confidence": ob.confidence, "image_quality_limitations": list(ob.quality_limitations)})
            self.sym_of.setdefault(ob.feature_id, []).append(sid)
        merged = observations_from(kb, Query(observed=r.reported_features, answers=r.answers))
        for fid in sorted(f for f, v in merged.items() if v and kb.features[f]["category"] != "environment"):
            n += 1
            sid = f"SYM-{n}"
            channel = r.reported_channels.get(fid, "symptom_checklist") if fid in r.reported_features else "guided_question_answer"
            via = {"symptom_checklist": "selected from the symptom list", "guided_question_answer": "answer to a guided question",
                   "typed_text_confirmed": "typed description, confirmed by the user",
                   "voice_transcript_confirmed": "voice transcript, confirmed by the user"}[channel]
            self.sym.append({
                "symptom_id": sid, "symptom_type": SYMPTOM_TYPE.get(fid, "other"), "label": self.L(kb.features[fid]["label"]),
                "evidence_status": "observed", "origin": "user_report", "input_channel": channel,
                "location": {"plant_part": PLANT_PART.get(fid, "leaf_unspecified"), "bbox": None},
                "affected_area_fraction": None, "measurement_method": None, "visual_evidence": self.L(f"Reported by the user ({via})."),
                "confidence": "not_assessable", "image_quality_limitations": []})
            self.sym_of.setdefault(fid, []).append(sid)
        # observations used for retrieval: user statements win over automated image observations
        self.obs = {ob.feature_id: True for ob in r.image_observations}
        self.obs.update(merged)
        img_conflict = sorted({ob.feature_id for ob in r.image_observations if merged.get(ob.feature_id) is False})
        if img_conflict:
            self.state("The automated image analysis and your report disagree about: "
                       + "; ".join(kb.features[f]["label"] for f in img_conflict) + ". Your report was used for ranking.")
        env = sorted(f for f, v in self.obs.items() if v and kb.features[f]["category"] == "environment")
        if env:
            self.state("Reported growing conditions (not symptoms): " + "; ".join(kb.features[f]["label"] for f in env) + ".")
        absent = sorted(f for f, v in self.obs.items() if v is False)
        if absent:
            self.state("Reported as absent: " + "; ".join(kb.features[f]["label"] for f in absent) + ".")
        if not r.image_observations:
            self.state("Automated image symptom analysis is not available in this version; listed symptoms come only from your "
                       "report and answers. Blur and exposure of the photo were not assessed.")

    def add_typical_symptoms(self, cid: str):
        """Source-described typical symptoms of the model's class that were NOT observed (knowledge_base_typical)."""
        c = self.kb.conditions[cid]
        n = len(self.sym)
        for f in c["features"]:
            if f["role"] not in ("distinguishing", "typical") or self.obs.get(f["feature_id"]):
                continue
            n += 1
            sid = f"SYM-{n}"
            self.sym.append({
                "symptom_id": sid, "symptom_type": SYMPTOM_TYPE.get(f["feature_id"], "other"),
                "label": self.L(f"Typical of {self.cname(cid)} according to the source (not observed in this plant): {f['claim']['text']}"),
                "evidence_status": "unverified", "origin": "knowledge_base_typical", "input_channel": "knowledge_base",
                "location": {"plant_part": PLANT_PART.get(f["feature_id"], "leaf_unspecified"), "bbox": None},
                "affected_area_fraction": None, "measurement_method": None, "visual_evidence": None,
                "confidence": "not_assessable", "image_quality_limitations": []})
            self.cite(f"visible_symptoms/{sid}", cid, f["claim"], "typical_symptom_described")

    # ---- recommendations ------------------------------------------------------------------------------------------
    def _geo_of_claims(self, claims):
        sids = {e["source_id"] for c in claims for e in c["evidence"]}
        return ({self.kb.sources[s]["geographic_relevance"] for s in sids},
                {self.kb.sources[s]["publisher_or_author"] for s in sids})

    def diag_test(self, did: str, cid: str | None) -> str:
        if did in self.diag_rec:
            return self.diag_rec[did]
        d = self.kb.diagnostics[did]
        rid = self.next_rec()
        self.diag_rec[did] = rid
        sids = sorted({s for c in d["claims"] for s in self.cite(f"recommended_tests/{rid}", did, c)})
        geos, pubs = self._geo_of_claims(d["claims"])
        rec = {"recommendation_id": rid, "category": "diagnostic_test", "subcategory": policy.DIAG_SUBCATEGORY[d["method_type"]],
               "text": self.L(f"{d['name']}: {d['description']}"), "consequential": False, "source_ids": sids,
               "applies_to": {"crop": self.pname(self.pid) if self.pid else None, "disease_id": cid, "region": None},
               "geographic_applicability": policy.geographic_applicability(geos), "dose": None,
               "what_it_can_establish": self.L(d["what_it_can_establish"]),
               "what_it_cannot_establish": self.L(d["what_it_cannot_establish"]), "expert_review_status": "pending"}
        risk = []
        if d["curation_status"] != "sourced":
            risk.append(self.L("The procedure is not yet backed by a verified source in the knowledge base."))
        if rec["geographic_applicability"] == "international_not_verified_for_India":
            risk.append(self.L(policy.FOREIGN_NOTE))
            self.warn("warn_foreign")
        if risk:
            rec["risk_notes"] = risk
        self.recs["recommended_tests"].append(rec)
        return rid

    def management(self, cid: str, depends_on: list[str]):
        kb = self.kb
        c = kb.conditions[cid]
        for mid in c["management_ids"]:
            m = kb.management[mid]
            if m["category"] == "nutrient":
                # Owner decision 2026-10-14: named corrective products are not given; recommend testing + expert advice.
                self.withheld.append({"management_id": mid, "condition_id": cid, "category": "nutrient",
                                      "reason": policy.NUTRIENT_PRODUCT_REASON,
                                      "source_ids": sorted({e["source_id"] for e in m["claim"]["evidence"]})})
                self.nutrient_testing(cid, depends_on, m)
                continue
            if m["category"] in policy.WITHHELD_CATEGORIES:
                ok, why = policy.chemical_decision(m, kb, self.verifications)
                if not ok:
                    self.withheld.append({"management_id": mid, "condition_id": cid, "category": m["category"], "reason": why,
                                          "source_ids": sorted({e["source_id"] for e in m["claim"]["evidence"]})})
                    continue
                raise ReportEngineError([f"{mid}: actionable chemical output is not implemented in this milestone"])
            if mid in self.mgmt_rec:
                continue
            section, cat, sub = policy.MGMT_MAP[m["category"]]
            rid = self.next_rec()
            self.mgmt_rec[mid] = rid
            sids = self.cite(f"{section}/{rid}", mid, m["claim"])
            geos, pubs = self._geo_of_claims([m["claim"]])
            geo = policy.geographic_applicability(geos)
            name = self.cname(cid)
            if cat == "suspected_nutrient_issue":
                text = (f"Suspected only, not confirmed. If a soil or plant-tissue test confirms {name.lower()}, the source describes "
                        f"this correction: {m['summary']} No rate is given here; ask the KVK or a soil-testing laboratory.")
            else:
                text = f"If {name} is confirmed, the source describes: {m['summary']}"
            risk = []
            if geo == "international_not_verified_for_India":
                text += " (Source from outside India; not verified for India.)"
                risk.append(self.L(policy.FOREIGN_NOTE))
                self.warn("warn_foreign")
            rn = policy.region_note(pubs, geos, self.req.region.state)
            if rn:
                risk.append(self.L(rn))
            rec = {"recommendation_id": rid, "category": cat, "subcategory": sub, "text": self.L(text),
                   "consequential": bool(m["consequential"]), "source_ids": sids,
                   "applies_to": {"crop": self.pname(self.pid) if self.pid else None, "disease_id": cid,
                                  "region": "India" if geo == "India_specific" else None},
                   "geographic_applicability": geo, "dose": None, "depends_on": list(depends_on), "expert_review_status": "pending"}
            if risk:
                rec["risk_notes"] = risk
            if m.get("safety_notes"):
                rec["safety_notes"] = [self.L(s) for s in m["safety_notes"]]
            if cat == "suspected_nutrient_issue":
                self.warn("warn_nutrient")
                rec["verification_test"] = self.L("Confirm the deficiency with a soil and/or plant-tissue nutrient analysis by a "
                                                  "qualified laboratory before any correction. (The knowledge base has no verified "
                                                  "source for the test procedure yet.)")
            self.recs[section].append(rec)

    def nutrient_testing(self, cid: str, depends_on: list[str], m: dict):
        """Suspected nutrient problem: testing + consultation only; no product, rate or soil value (one per condition)."""
        if cid in self.nutrient_rec:
            return
        rid = self.next_rec()
        self.nutrient_rec[cid] = rid
        self.warn("warn_nutrient")
        rec = {"recommendation_id": rid, "category": "suspected_nutrient_issue", "subcategory": "laboratory_test",
               "text": self.T("rec_nutrient_test", condition=self.cname(cid)), "consequential": False, "source_ids": [],
               "applies_to": {"crop": self.pname(self.pid) if self.pid else None, "disease_id": cid, "region": None},
               "geographic_applicability": "general", "dose": None, "depends_on": list(depends_on), "expert_review_status": "pending",
               "verification_test": self.T("rec_nutrient_verification")}
        if m.get("safety_notes"):
            rec["safety_notes"] = [self.L(s) for s in m["safety_notes"]]
        self.recs["nutrition_guidance"].append(rec)

    def fixed_action(self, key: str, category: str, subcategory: str):
        rid = self.next_rec()
        self.recs["immediate_actions"].append({
            "recommendation_id": rid, "category": category, "subcategory": subcategory, "text": self.T(key), "consequential": False,
            "source_ids": [], "applies_to": {"crop": self.pname(self.pid) if self.pid else None, "disease_id": None, "region": None},
            "geographic_applicability": "general", "dose": None, "expert_review_status": "pending"})

    # ---- candidates -----------------------------------------------------------------------------------------------
    def candidate_from_condition(self, cand_id: str, cid: str, *, method: str, score, sup_feats: list, con_feats: list,
                                 missing: list) -> dict:
        kb = self.kb
        c = kb.conditions[cid]
        p = f"candidate_diagnoses/{cand_id}"
        sids: set = set()
        hosts = [a["plant_id"] for a in c["affected_plants"]]
        for a in c["affected_plants"]:
            sids |= set(self.cite(p + "/affected_crop", cid, a["claim"]))
        sc = {"value": None, "status": "not_assessable", "source_ids": []}
        if c["scientific_cause"]:
            ids = self.cite(p + "/scientific_cause", cid, c["scientific_cause"]["claim"])
            sc = {"value": c["scientific_cause"]["name"], "status": "unverified", "source_ids": ids}
            sids |= set(ids)
        links = {f["feature_id"]: f for f in c["features"] + c["favourable_conditions"]}
        supporting = []
        for fid in sup_feats:
            sids |= set(self.cite(p + "/supporting_symptoms", cid, links[fid]["claim"]))
            supporting += self.sym_of.get(fid, [])
        conflicts = []
        for fid in con_feats:
            sids |= set(self.cite(p + "/conflicting_or_missing", cid, links[fid]["claim"], "conflict"))
            conflicts.append(self.L(f"Conflicting: you reported '{kb.features[fid]['label']}' as absent, but the source describes it "
                                    f"as {links[fid]['role']} for {self.cname(cid)}."))
        for fid in missing:
            conflicts.append(self.L(f"Not yet checked: '{kb.features[fid]['label']}' ({links[fid]['role']} for {self.cname(cid)} "
                                    "according to the source)."))
        alts = []
        for la in c["look_alikes"]:
            t = f"{self.cname(la['condition_id'])} (look-alike: {la['basis'].replace('_', ' ')})"
            if la.get("claim"):
                sids |= set(self.cite(p + "/important_alternatives", cid, la["claim"]))
                t += f": {la['claim']['text']}"
            alts.append(self.L(t))
        tests = [self.diag_test(d, cid) for d in c["diagnostic_ids"]]
        compatible = bool(self.confirmed and self.pid and self.pid in hosts)
        title = self.cname(cid) if compatible else f"{self.cname(cid)} (recorded host: {', '.join(self.pname(h) for h in hosts)})"
        cand = {"candidate_id": cand_id, "status": "candidate", "kb_disease_id": cid, "candidate_name": self.L(title),
                "cause_category": CAUSE.get(c["cause_category"], c["cause_category"]), "scientific_cause": sc,
                "affected_crop": ", ".join(self.pname(h) for h in hosts), "crop_compatible": compatible,
                "supporting_symptoms": sorted(set(supporting), key=lambda s: int(s.split("-")[1])),
                "conflicting_or_missing": conflicts, "reference_image_matches": [], "important_alternatives": alts,
                "recommended_confirmatory_tests": tests, "source_ids": sorted(sids), "ranking_method": method,
                "ranking_score": score, "ranking_score_is_probability": False, "validation_status": "unvalidated",
                "uncertainty_statement": self.T({"model_class_evidence": "candidate_model_class_uncertainty",
                                                 "hybrid": "candidate_hybrid_uncertainty"}.get(method, "candidate_uncertainty"))}
        if not compatible:
            cand["crop_mismatch_note"] = self.T("candidate_preliminary" if not (self.confirmed and self.pid) else "candidate_crop_mismatch")
        if c["cause_category"] in ("nutrient_deficiency", "nutrient_toxicity"):
            self.warn("warn_nutrient")
        if "international_not_verified_for_India" in c["geographic_applicability"] and not any(
                g.startswith("India") for g in c["geographic_applicability"]):
            self.warn("warn_foreign")
            self.state(f"{self.cname(cid)}: the knowledge-base evidence comes only from sources outside India "
                       "(not verified for India).")
        # coverage registry: knowledge vs image-diagnosis capability
        cv = self.cov["conditions"][cid]
        img = cv["image_diagnosis"]
        txt = (f"Coverage for {self.cname(cid)}: knowledge {cv['knowledge_base_information']} ({len(cv['sources_verified'])} verified "
               f"source(s)); image diagnosis: {_pct(img['note'])}")
        ref = img.get("frozen_stage8_osr_reference")
        if ref:
            txt += (f". Research reference (Stage 8, {ref['detector']}): AUROC {ref['auroc_vs_known']}, fraction of that disease's "
                    f"research test images flagged as unfamiliar at the frozen threshold {ref['detection_at_frozen_threshold']}")
        txt += f"; expert reviewed: {'yes' if cv['expert_reviewed'] else 'no'}."
        self.state(txt)
        return cand

    def from_retrieval(self, rc: dict, cand_id: str) -> dict:
        env = {f for f, x in self.kb.features.items() if x["category"] == "environment"}
        sup = [s["feature_id"] for s in rc["supporting_evidence"]]
        con = [s["feature_id"] for s in rc["conflicting_evidence"]]
        missing = [m["feature_id"] for m in rc["missing_information"] if m["feature_id"] not in env]
        return self.candidate_from_condition(cand_id, rc["condition_id"], method="kb_symptom_match", score=rc["ranking_score"],
                                             sup_feats=sup, con_feats=con, missing=missing)

    def model_class_candidate(self, cid: str) -> dict:
        c = self.kb.conditions[cid]
        sup, con, missing = [], [], []
        for f in c["features"] + c["favourable_conditions"]:
            v = self.obs.get(f["feature_id"])
            if v:
                sup.append(f["feature_id"])
            elif v is False and f in c["features"] and f["role"] in ("distinguishing", "typical"):
                con.append(f["feature_id"])
            elif v is None and f in c["features"] and f["role"] in ("distinguishing", "typical"):
                missing.append(f["feature_id"])
        return self.candidate_from_condition("CAND-1", cid, method="model_class_evidence", score=None, sup_feats=sup,
                                             con_feats=con, missing=missing)

    def hybrid_candidate(self, cand_id: str, cid: str, score: float, rc: dict | None) -> dict:
        """Candidate from the model's closest classes (photo not accepted) combined with the sign match (SC-M5-02)."""
        c = self.kb.conditions[cid]
        env = {f for f, x in self.kb.features.items() if x["category"] == "environment"}
        if rc:
            sup = [s["feature_id"] for s in rc["supporting_evidence"]]
            con = [s["feature_id"] for s in rc["conflicting_evidence"]]
            missing = [m["feature_id"] for m in rc["missing_information"] if m["feature_id"] not in env]
        else:
            sup = [f["feature_id"] for f in c["features"] + c["favourable_conditions"] if self.obs.get(f["feature_id"])]
            con = [f["feature_id"] for f in c["features"] if self.obs.get(f["feature_id"]) is False and f["role"] in ("distinguishing", "typical")]
            missing = [f["feature_id"] for f in c["features"] if self.obs.get(f["feature_id"]) is None and f["role"] in ("distinguishing", "typical")]
        return self.candidate_from_condition(cand_id, cid, method="hybrid", score=score, sup_feats=sup, con_feats=con, missing=missing)

    def differential(self, cands: list, questions: list) -> dict:
        kb = self.kb
        rows = []
        if not cands:
            return {"rows": rows}
        env = {f for f, x in kb.features.items() if x["category"] == "environment"}

        def row(finding, cells, label=None):
            r = {"finding": finding, "per_candidate": cells}
            if label:
                r["finding_label"] = self.L(label)
            rows.append(r)
        cells = {}
        for c in cands:
            hosts_src = sorted({e["source_id"] for a in kb.conditions[c["kb_disease_id"]]["affected_plants"] for e in a["claim"]["evidence"]})
            if c["crop_compatible"]:
                cells[c["candidate_id"]] = {"assessment": "supports", "note": self.L(f"Recorded on {c['affected_crop']}."), "source_ids": hosts_src}
            else:
                cells[c["candidate_id"]] = {"assessment": "unknown" if not self.confirmed else "conflicts",
                                            "note": c["crop_mismatch_note"], "source_ids": hosts_src}
        row("crop_compatibility", cells)
        sym_by_id = {s["symptom_id"]: s for s in self.sym}
        cells = {}
        for c in cands:
            labels = [sym_by_id[s]["label"]["en"] for s in c["supporting_symptoms"]]
            cells[c["candidate_id"]] = {"assessment": "supports" if labels else "neutral",
                                        "note": self.L("Explains: " + "; ".join(labels) if labels else "No observed symptom is explained by the source description."),
                                        "source_ids": c["source_ids"] if labels else []}
        row("observed_symptoms_explained", cells)
        cells = {}
        for c in cands:
            conf = [x["en"] for x in c["conflicting_or_missing"] if x["en"].startswith("Conflicting")]
            miss = [x["en"] for x in c["conflicting_or_missing"] if x["en"].startswith("Not yet checked")]
            a = "conflicts" if conf else ("unknown" if miss else "neutral")
            cells[c["candidate_id"]] = {"assessment": a, "note": self.L(" ".join(conf + miss) or "No expected sign is missing or contradicted."),
                                        "source_ids": c["source_ids"] if conf else []}
        row("missing_expected_signs", cells)
        cells = {}
        for c in cands:
            cond = kb.conditions[c["kb_disease_id"]]
            hit = [f for f in cond["favourable_conditions"] if self.obs.get(f["feature_id"])]
            cells[c["candidate_id"]] = {"assessment": "supports" if hit else "unknown",
                                        "note": self.L("; ".join(f["claim"]["text"] for f in hit) if hit else "No matching growing condition reported."),
                                        "source_ids": sorted({e["source_id"] for f in hit for e in f["claim"]["evidence"]})}
        row("environmental_compatibility", cells)
        cells = {}
        for c in cands:
            ds = [kb.diagnostics[d] for d in kb.conditions[c["kb_disease_id"]]["diagnostic_ids"]]
            cells[c["candidate_id"]] = {"assessment": "neutral",
                                        "note": self.L("; ".join(d["name"] for d in ds) if ds else "No diagnostic method recorded."),
                                        "source_ids": sorted({e["source_id"] for d in ds for cl in d["claims"] for e in cl["evidence"]})}
        row("distinguishing_test", cells)
        for q in questions:
            cells = {}
            for c in cands:
                cond = kb.conditions[c["kb_disease_id"]]
                links = {f["feature_id"]: f for f in cond["features"] + cond["favourable_conditions"]}
                hits = [(o["label"], f) for o in kb.questions[q["question_id"]]["options"] for f, v in o["sets"].items() if v and f in links]
                if hits:
                    cells[c["candidate_id"]] = {"assessment": "unknown",
                                                "note": self.L("Answer " + "; ".join(f"'{lab}'" for lab, _ in hits) + " would support this candidate."),
                                                "source_ids": sorted({e["source_id"] for _, f in hits for e in links[f]["claim"]["evidence"]})}
                else:
                    cells[c["candidate_id"]] = {"assessment": "neutral", "note": self.L("Not described for this candidate."), "source_ids": []}
            row("other", cells, label=f"Unanswered question: {q['text']}")
        _ = env
        return {"rows": rows}

    # ---- language-independent content keys (D-6) ------------------------------------------------------------------
    def content_keys(self, report: dict) -> list:
        out = []

        def walk(o, path):
            if isinstance(o, dict):
                if id(o) in self.keyed:
                    key, params = self.keyed[id(o)]
                    out.append({"path": path, "key": key, "params": params})
                    return
                for k, v in o.items():
                    if k != "content_keys":
                        walk(v, f"{path}/{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}/{i}")
        walk(report, "")
        return out

    # ---- evidence sources -----------------------------------------------------------------------------------------
    def evidence_sources(self, report: dict) -> list:
        used = set()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "source_ids":
                        used.update(v)
                    elif k == "source_id" and isinstance(v, str):
                        used.add(v)
                    else:
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(report)
        used |= {s for w in self.withheld for s in w["source_ids"]}
        out = []
        for sid in sorted(used):
            s = self.kb.sources[sid]
            claims = sorted({t["claim"] for t in self.trace if any(e["source_id"] == sid for e in t["evidence"])})
            claims += sorted({f"Lists {w['category']} options for {self.cname(w['condition_id'])} (withheld from this report: "
                              f"{w['reason']})." for w in self.withheld if sid in w["source_ids"]})
            out.append({"source_id": sid, "title": s["title"], "publisher_or_author": s["publisher_or_author"], "url_or_doi": s["url"],
                        "published_or_updated": s["published_or_updated"], "accessed_on": s["accessed_on"],
                        "claims_supported": claims, "geographic_relevance": SOURCE_GEO[s["geographic_relevance"]],
                        "source_type": s["source_type"],
                        "limitations": f"{s['limitations']} | knowledge-base geography: {s['geographic_relevance']} | review status: "
                                       f"{s['review_status']} (not expert-reviewed)" if s["review_status"] != "expert_reviewed" else s["limitations"],
                        "license_or_terms": s["license_or_terms"]})
        return out


HYBRID_W_MODEL = 0.6          # a-priori weight of the model's softmax in the hybrid ranking (SC-M5-02); not tuned


def hyb_model_ids(b: "_Builder") -> set:
    return set(b.hyb_model)


def _hybrid_ranking(b: "_Builder", rr: dict) -> list | None:
    """For a confirmed in-scope crop whose photo the frozen model did NOT accept: rank sourced conditions of that crop by
    HYBRID_W_MODEL * softmax (model top-3 classes mapped to the KB) + (1 - HYBRID_W_MODEL) * retrieval score scaled to
    the best retrieval score. 'Healthy' classes and classes of other crops are skipped. None when the model did not run."""
    a = b.req.model_assessment
    b.hyb_model = {}
    if a is None or not (b.confirmed and b.pid and b.in_scope):
        return None
    cls2cond = {m["class_name"]: m["condition_id"] for m in b.kb.raw["model_classes"]}
    for t in a["model_prediction"]["top3"]:
        cid = cls2cond.get(t["class"])
        if not cid or b.kb.conditions[cid]["curation_status"] != "sourced":
            continue
        if b.pid not in {x["plant_id"] for x in b.kb.conditions[cid]["affected_plants"]}:
            continue
        # the user's own report wins: a distinguishing sign reported as absent rules out a model-only suggestion
        if any(f["role"] == "distinguishing" and b.obs.get(f["feature_id"]) is False for f in b.kb.conditions[cid]["features"]):
            continue
        b.hyb_model[cid] = max(b.hyb_model.get(cid, 0.0), float(t["softmax"]))
    rmax = max([rc["ranking_score"] for rc in rr["candidates"]] or [0.0])
    by_ret = {rc["condition_id"]: rc for rc in rr["candidates"]}
    score = {}
    for cid in set(b.hyb_model) | set(by_ret):
        r = by_ret[cid]["ranking_score"] / rmax if cid in by_ret and rmax > 0 else 0.0
        score[cid] = round(HYBRID_W_MODEL * b.hyb_model.get(cid, 0.0) + (1 - HYBRID_W_MODEL) * r, 4)
    ranked = sorted(score, key=lambda c: (-score[c], c))
    return [(c, score[c], by_ret.get(c)) for c in ranked if score[c] > 0] or None


def _report_type(b: _Builder) -> str:
    if b.req.image.quality_issues:
        return "insufficient_image"
    if b.confirmed and not b.in_scope:
        return "out_of_model_scope"
    if b.osr_decision == "known" and b.in_scope:
        return "known_disease"
    return "unknown_investigation"


def build_report(request: ReportRequest, kb: KnowledgeBase | None = None, *, verifications: list | None = None) -> ReportResult:
    from stage10.report_engine.checks import engine_errors

    kb = kb or load_kb()
    req = validate_request(request, kb)
    errs = check_model_assessment(req.model_assessment) if req.model_assessment else []
    if errs:
        raise ReportEngineError(errs)
    b = _Builder(req, kb, build_coverage(kb))
    b.verifications = policy.load_verifications() if verifications is None else verifications
    b.resolve_plant()
    rtype = _report_type(b)
    identity = b.plant_identity()
    b.symptoms()
    max_c = b.cfg["candidates"]["max_candidates"]
    cands, questions, insufficient = [], [], None

    if rtype == "insufficient_image":
        insufficient = b.T("insufficient_image")
        b.state("The image failed the basic quality gate: " + "; ".join(req.image.quality_issues) + ". No image-based result is interpreted.")
    else:
        plant_q = b.pname(b.pid) if b.pid else (req.plant_name if b.confirmed else None)
        rr = retrieve(Query(plant=plant_q, observed=b.obs, region=req.region.state), kb)
        if rtype == "known_disease":
            mc = next((m for m in kb.raw["model_classes"] if m["class_name"] == b.model_class), None)
            cid = mc["condition_id"] if mc else None
            mismatch = b.confirmed and b.model_crop != b.crop_label
            if mismatch:
                b.state(f"The model's class '{b.model_class}' belongs to {b.model_crop}, not to {b.pname(b.pid)} (your confirmed crop). "
                        "The model's disease class is therefore not applicable and is not presented as a candidate.")
            elif cid is None:
                b.state(f"The model's class '{b.model_class}' is a 'healthy' class. This does not establish that the plant is free "
                        "of disease, nutrient problems or pests.")
            elif kb.conditions[cid]["curation_status"] != "sourced":
                b.state(f"The model's class '{b.model_class}' maps to a knowledge-base record that is not yet source-verified "
                        f"({cid}). No disease information, symptoms or management can be given for it yet.")
            else:
                b.add_typical_symptoms(cid)
                cands.append(b.model_class_candidate(cid))
                c = kb.conditions[cid]
                b.state(f"Why {b.cname(cid)} is listed: the frozen model accepted the image as its known class '{b.model_class}' "
                        f"(contrastive Energy {req.model_assessment['osr_status']['score']:.3f} at or below the frozen threshold). "
                        "A model class is not a confirmed diagnosis.")
                if c["curability"]["status"] != "not_documented" and c["curability"].get("claim"):
                    b.cite(b.state(f"Treatability (source): {c['curability']['claim']['text']}"), cid, c["curability"]["claim"])
                else:
                    b.state(f"Treatability of {b.cname(cid)} is not documented in the verified sources.")
                for i, s in enumerate(c["spread_and_survival"]):
                    b.cite(b.state(f"Disease information (source): {s['text']}"), cid, s)
                for s in c.get("seasonality", []):
                    b.cite(b.state(f"Seasonality (source): {s['text']}"), cid, s)
            b.state("What the available evidence cannot establish: the model output does not confirm the disease, its severity, "
                    "or that other problems are absent. The model was evaluated only on laboratory-style research images and has "
                    "not been validated on field photographs or as an agricultural diagnostic.")
            for rc in rr["candidates"]:
                if len(cands) >= max_c:
                    break
                if all(rc["condition_id"] != c["kb_disease_id"] for c in cands):
                    cands.append(b.from_retrieval(rc, f"CAND-{len(cands) + 1}"))
            questions = suggest_questions(kb, [{"condition_id": c["kb_disease_id"]} for c in cands], b.obs,
                                          b.cfg["candidates"]["retrieval"]["max_questions"]) if cands else []
        else:
            if rr["status"] == "plant_not_in_knowledge_base":
                insufficient = b.T("insufficient_plant_not_in_kb")
            hyb = _hybrid_ranking(b, rr) if rtype == "unknown_investigation" else None
            if hyb:
                for n, (cid, score, rc) in enumerate(hyb[:max_c], 1):
                    cands.append(b.from_retrieval(rc, f"CAND-{n}") if rc and cid not in hyb_model_ids(b)
                                 else b.hybrid_candidate(f"CAND-{n}", cid, score, rc))
                b.state("Possible causes combine the frozen model's closest classes for this crop (the photo did not meet its "
                        "acceptance criterion, so none is a model result) with the knowledge-base match of the observed signs. "
                        f"Weights are fixed a priori ({HYBRID_W_MODEL:.1f} model similarity, {1 - HYBRID_W_MODEL:.1f} sign match), "
                        "not tuned on data; the order is unvalidated.")
            else:
                for rc in rr["candidates"][:max_c]:
                    cands.append(b.from_retrieval(rc, f"CAND-{len(cands) + 1}"))
            questions = rr["questions"]
            if not cands and insufficient is None:
                insufficient = b.T("insufficient_evidence")
            if rtype == "out_of_model_scope" and req.model_assessment is not None:
                b.state("The frozen image model was run, but it covers only Apple, Corn, Grape, Pepper bell, Potato and Tomato; "
                        "its class and known/unknown decision are not applicable to this plant.")
            if rtype == "unknown_investigation" and req.model_assessment is not None:
                b.state("The frozen model's known-class output is not used as a diagnosis because the image did not meet its "
                        "acceptance criterion.")
        if rtype == "known_disease" and not cands:
            insufficient = b.T("insufficient_evidence")
        hosts = {a["plant_id"] for c in cands for a in kb.conditions[c["kb_disease_id"]]["affected_plants"]}
        if not b.confirmed and len(hosts) > 1:
            b.state("The candidates concern different host plants. Confirm the plant before treating any of them as a "
                    "meaningful diagnosis; cross-crop matches are preliminary and not actionable.")
        if questions:
            b.state("Unanswered questions that would help distinguish the candidates: " + " | ".join(q["text"] for q in questions))

    # management only for candidates on a user-confirmed host crop
    for c in cands:
        if c["crop_compatible"]:
            b.management(c["kb_disease_id"], c["recommended_confirmatory_tests"])
    if b.withheld:
        b.state("Withheld from this report: " + "; ".join(
            f"{w['category']} options for {b.cname(w['condition_id'])} listed by {', '.join(w['source_ids'])} ({w['reason']})"
            for w in b.withheld) + ".")
    if cands and not any(c["crop_compatible"] for c in cands):
        b.state("No management guidance is given because no candidate's host crop has been confirmed.")
    if b.pid and not (b.recs["soil_guidance"] or b.recs["nutrition_guidance"]):
        b.state(f"No source-verified general crop-care (soil, nutrition, irrigation) guidance for {b.pname(b.pid)} is in the "
                f"knowledge base (v{kb.version}) yet.")
    if b.pid:
        pc = b.cov["plants"][b.pid]
        b.state(f"Plant coverage for {b.pname(b.pid)}: {len(pc['conditions_sourced'])} source-verified condition(s) in the knowledge "
                f"base; image diagnosis: {_pct(pc['image_diagnosis']['note'])}; geographic applicability of evidence: "
                f"{', '.join(pc['geographic_applicability']) or 'none'}; expert reviewed: no.")
    # fixed immediate actions
    if not b.confirmed:
        b.warn("warn_crop_unconfirmed")
        b.fixed_action("act_confirm_crop", "immediate_action", "other")
    if rtype == "insufficient_image":
        b.fixed_action("act_retake_photo", "immediate_action", "photography")
    elif rtype != "known_disease" or not cands:
        b.fixed_action("act_more_photos", "immediate_action", "photography")
    b.fixed_action("act_no_spray_yet", "immediate_action", "other")
    b.fixed_action("act_consult_expert", "escalation", "consult_expert")
    if b.withheld:
        b.fixed_action("act_registered_products", "escalation", "consult_expert")
    b.fixed_action("act_monitor", "follow_up", "monitoring")
    if b.confirmed and not b.in_scope:
        b.warn("warn_outside_model")
    if any(c["ranking_method"] == "kb_symptom_match" for c in cands):
        b.warn("warn_ranking")
    if b.osr_decision == "unknown":
        b.warn("warn_unfamiliar_not_new")
    if req.language == "hi":
        b.warn("translation_note")

    # model output: preserved verbatim; only localized texts gain a translation
    if req.model_assessment:
        mp = copy.deepcopy(req.model_assessment["model_prediction"])
        osr = copy.deepcopy(req.model_assessment["osr_status"])
        for key, src in (("message", MSG_KNOWN if osr["decision"] == "known" else MSG_UNKNOWN), ("interpretation_note", NOTE)):
            assert osr[key]["en"] == src["en"]
        if req.language == "hi":
            osr["message"]["hi"] = b.msg["osr_known" if osr["decision"] == "known" else "osr_unknown"]["hi"]
            osr["interpretation_note"]["hi"] = b.msg["osr_note"]["hi"]
        b.keyed[id(osr["message"])] = ("osr_known" if osr["decision"] == "known" else "osr_unknown", {})
        b.keyed[id(osr["interpretation_note"])] = ("osr_note", {})
    else:
        mp = {"ran": False, "predicted_class": None, "predicted_crop": None, "top3": [], "softmax_is_calibrated": False,
              "model_versions": dict(frozen_model_versions())}
        osr = {"decision": "not_run", "score_name": "contrastive_energy", "score": None, "frozen_threshold": frozen_threshold(),
               "rule": RULE, "percentile_known_validation": None, "borderline_display": False,
               "message": b.T("osr_not_run"), "interpretation_note": b.T("osr_note")}

    has_conflict = any(x["en"].startswith("Conflicting") for c in cands for x in c["conflicting_or_missing"])
    if rtype == "known_disease" and cands and cands[0]["ranking_method"] == "model_class_evidence" and cands[0]["crop_compatible"] \
            and cands[0]["supporting_symptoms"] and not has_conflict:
        level = "moderate"
    elif cands:
        level = "low"
    else:
        level = "not_assessable" if rtype != "known_disease" else "low"

    # report_id is language-independent: the same case rendered in English or Hindi keeps one identity (D-6)
    canon = json.dumps({"request": {k: v for k, v in req.canonical().items() if k != "language"}, "kb": kb.version,
                        "config": configuration_fingerprint()},
                       sort_keys=True, ensure_ascii=False, default=list)
    report = {
        "schema_version": b.cfg["schema_version"],
        "report_id": f"AV-{req.generated_at[:10].replace('-', '')}-{_sha(canon.encode('utf-8'))[:12]}",
        "created_at": req.generated_at, "report_type": rtype, "language": req.language,
        "region": {"country": "IN", "state": req.region.state, "district": req.region.district},
        "image_metadata": {"sha256": req.image.sha256, "width": req.image.width, "height": req.image.height, "format": req.image.format,
                           "quality_gate": {"passed": not req.image.quality_issues, "issues": list(req.image.quality_issues)}},
        "plant_identity": identity, "model_prediction": mp, "osr_status": osr, "visible_symptoms": b.sym,
        "candidate_diagnoses": cands, "differential_diagnosis": b.differential(cands, questions),
        **b.recs,
        "follow_up_plan": {"enabled": False, "suggested_interval_days": None,
                           "what_to_record": [b.T(k) for k in ("record_photo", "record_spread", "record_weather", "record_tests")]},
        "evidence_sources": [],
        "uncertainty": {"level": level, "statements": b.statements},
        "safety_warnings": [b.T(k) for k in b.warn_keys], "disclaimer": b.T("disclaimer"),
        "review_status": {"expert_review": "pending",
                          "translation_review": {"en": "not_applicable", **({"hi": "pending_fluent_speaker_review"} if req.language == "hi" else {})}},
        "generator": {"name": ENGINE_NAME, "version": ENGINE_VERSION, "mode": "deterministic_template", "llm_used": False,
                      "knowledge_base_version": kb.version},
    }
    if insufficient is not None:
        report["insufficient_evidence_statement"] = insufficient
    report["evidence_sources"] = b.evidence_sources(report)
    report["content_keys"] = b.content_keys(report)
    trace = {"report_id": report["report_id"], "knowledge_base_version": kb.version, "engine_version": ENGINE_VERSION,
             "entries": b.trace, "withheld_management": b.withheld,
             "note": "Each entry links a report element to the knowledge-base claim it uses and that claim's verbatim source passages."}
    errs = engine_errors(report, trace, kb, req)
    if errs:
        raise ReportEngineError(errs)
    return ReportResult(report, trace)
