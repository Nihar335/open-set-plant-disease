"""Plant-health assessment from a USER-PROVIDED plant name (product-scope decision SC-M4-03).

Flow: confirmed plant-name choice (PlantNameSession.confirm()) -> routing -> frozen disease model ONLY for a confirmed,
knowledge-base plant inside its six-crop scope, through the existing FrozenOSRAdapter.assess() -> M3 report engine.

Guarantees:
  * automatic plant identification (BioCLIP 2) is never imported or called; the plant name is the user's;
  * the disease model is NOT invoked for an unsupported plant, an unlisted name, an unknown plant or an image that fails
    the quality gate; the report then says the model did not run;
  * a disease-model prediction never confirms the user's plant name (the engine reports agreement as supporting evidence
    only, and a crop mismatch makes the model class not applicable);
  * user-reported symptoms keep their input channel; image-derived observations stay labelled as inferred by the engine;
  * the report engine's own validation (schema, evidence trace, dose and chemical restrictions) runs unchanged.
"""
from __future__ import annotations

from pathlib import Path

from stage10.plant_id.routing import route
from stage10.report_engine.engine import build_report
from stage10.report_engine.inputs import ImageInfo, Region, ReportRequest, image_info_from_file
from stage10.schemas.validate import structural_errors

WORKFLOW_VERSION = "1.0.0"
PLANT_ID_STATUS = ("disabled: automatic plant identification is not used (protocol v2 and v3 development thresholds unresolved; "
                   "product-scope decision SC-M4-03). The plant name is supplied by the user.")


class AssessmentError(ValueError):
    pass


def routing_decision(choice: dict, kb, image: ImageInfo) -> dict:
    """Which components run, and why. Pure: does not load any model."""
    if not isinstance(choice, dict) or choice.get("source") != "user_supplied" or "outcome" not in choice:
        raise AssessmentError("a confirmed plant-name choice from PlantNameSession.confirm() is required")
    outcome = choice["outcome"]
    if outcome == "known_plant" and choice["plant_id"] not in kb.plants:
        raise AssessmentError("confirmed plant id is not in the knowledge base")
    confirmed = outcome in ("known_plant", "unlisted_name")
    name = choice["plant_name"] if confirmed else None
    r = route(name, "confirmed" if confirmed else "not_provided", kb, None)
    supported = outcome == "known_plant" and bool(kb.plants[choice["plant_id"]]["frozen_model_scope"])
    if outcome == "known_plant" and r["kb_plant_id"] != choice["plant_id"]:
        raise AssessmentError("routing resolved a different plant than the user confirmed")
    if not supported:
        invoke, reason = False, {"known_plant": "the confirmed plant is outside the disease model's six crops: knowledge-based guidance",
                                 "unlisted_name": "the plant is not in the knowledge base: general guidance and expert referral",
                                 "not_known": "the plant is not known: the disease model needs a confirmed supported crop"}[outcome]
    elif image.quality_issues:
        invoke, reason = False, "the image failed the quality gate: retake the photo"
    else:
        invoke, reason = True, "the confirmed plant is one of the disease model's six crops"
    return {"plant_confirmation": "confirmed" if confirmed else "not_provided", "plant_name_for_report": name,
            "kb_plant_id": r["kb_plant_id"], "disease_model_supported": supported, "invoke_disease_model": invoke,
            "reason": reason, "knowledge_base_mode": r["knowledge_base"], "frozen_role": r["frozen_disease_model"],
            "automatic_plant_identification": PLANT_ID_STATUS}


def assess_plant_health(choice: dict, image_path, kb, *, reported_features: dict | None = None, reported_channels: dict | None = None,
                        answers: dict | None = None, region: Region = Region(), language: str = "en",
                        generated_at: str | None = None, osr_adapter=None, image_info: ImageInfo | None = None,
                        image_observations: tuple = ()) -> dict:
    """Run the routed assessment and return {workflow, report, evidence_trace}.

    image_observations: ImageObservation records from the visible-sign analyser (stage10.vision.symptoms), labelled
    'inferred' by the engine. osr_adapter: an object with assess(image) shaped like FrozenOSRAdapter (created lazily, and only when the routing says
    the disease model runs). image_info: precomputed metadata (tests); otherwise read from image_path with the data guard.
    """
    image = image_info or image_info_from_file(image_path)
    dec = routing_decision(choice, kb, image)
    assessment = None
    if dec["invoke_disease_model"]:
        if osr_adapter is None:
            from stage10.osr.osr_adapter import FrozenOSRAdapter
            osr_adapter = FrozenOSRAdapter()
        assessment = osr_adapter.assess(Path(image_path) if image_path is not None else None)
    req = ReportRequest(image=image, plant_name=dec["plant_name_for_report"], plant_confirmation=dec["plant_confirmation"],
                        reported_features=dict(reported_features or {}), reported_channels=dict(reported_channels or {}),
                        answers=dict(answers or {}), image_observations=tuple(image_observations),
                        model_assessment=assessment, region=region, language=language,
                        generated_at=generated_at)
    res = build_report(req, kb)
    rep = res.report
    # invariants of this workflow on top of the engine's own checks
    if rep["model_prediction"]["ran"] != dec["invoke_disease_model"]:
        raise AssessmentError("report disagrees with the routing about whether the disease model ran")
    if not dec["disease_model_supported"] and any(c["ranking_method"] == "model_class_evidence" for c in rep["candidate_diagnoses"]):
        raise AssessmentError("a disease-model diagnosis appeared for a plant outside the model's scope")
    wf = {"workflow_version": WORKFLOW_VERSION,
          "plant_selection": {k: choice[k] for k in ("outcome", "plant_id", "plant_name", "entered_text", "matched_alias", "alias_kind",
                                                     "entry_channel", "source", "independently_verified",
                                                     "automatic_identification_used", "corrections")},
          "routing": {k: dec[k] for k in ("disease_model_supported", "invoke_disease_model", "reason", "knowledge_base_mode",
                                          "automatic_plant_identification")},
          "disease_model_ran": rep["model_prediction"]["ran"], "report_id": rep["report_id"], "report_type": rep["report_type"]}
    err = structural_errors(wf, "plant_name_assessment")
    if err:
        raise AssessmentError(err)
    return {"workflow": wf, "report": rep, "evidence_trace": res.evidence_trace}
