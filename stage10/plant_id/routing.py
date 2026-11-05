"""Model routing for Milestone 4 (M4-08): which component may run for a case, and in what role.

Authorised behaviour only (PROPOSAL.md section 1):
  * the plant identity used by reports is ALWAYS the user-confirmed plant; a plant-identification result is a suggestion
    shown for confirmation and never changes the confirmation;
  * the frozen six-crop disease model is applicable only to a user-confirmed plant inside its scope. For a confirmed plant
    outside its scope it is 'not_applicable' (an out-of-scope report, with knowledge-base guidance where available),
    however confident any identification is;
  * with no confirmed plant, the frozen model's output is unconfirmed-scope evidence only (the M3 engine behaviour), and
    an 'unidentified' identification leads to an investigation report;
  * plant identification never produces a disease assessment.
Library code only; not wired into a farmer-facing application (integration is behind its explicit approval gate).
"""
from __future__ import annotations

FROZEN_ROLE = {"run": "disease model applicable (confirmed plant in its six-crop scope)",
               "not_applicable": "disease model not applicable to the confirmed plant",
               "unconfirmed_scope": "disease model output is evidence only, because the plant is not confirmed"}


def route(plant_name: str | None, plant_confirmation: str, kb, plant_id_result: dict | None = None) -> dict:
    if plant_confirmation not in ("confirmed", "not_provided", "unsure"):
        raise ValueError("plant_confirmation must be confirmed / not_provided / unsure")
    confirmed = plant_confirmation == "confirmed" and bool(plant_name)
    pid = kb.resolve_plant(plant_name) if confirmed else None          # same resolution as the M3 report engine
    in_scope = bool(pid and kb.plants[pid]["frozen_model_scope"])
    if plant_id_result is None:
        pid_role = "not_run"
    elif plant_id_result.get("sets_confirmed_crop") or not plant_id_result.get("requires_user_confirmation", False):
        raise ValueError("plant-identification result violates the contract (must be a suggestion requiring confirmation)")
    else:
        pid_role = "suggestion_only" if plant_id_result["status"] == "candidates" else f"refused:{plant_id_result['status']}"
    if confirmed:
        frozen = "run" if in_scope else "not_applicable"
        kb_mode = "plant_specific" if pid else "general_investigation"
        expected = None if in_scope else "out_of_model_scope"   # unless the image itself is insufficient
    else:
        frozen, kb_mode = "unconfirmed_scope", "general_investigation"
        expected = None                     # decided by the M3 engine from image quality and the frozen OSR result
    return {"identity_source": "user_confirmed" if confirmed else "unconfirmed",
            "confirmed_plant": plant_name if confirmed else None, "kb_plant_id": pid,
            "plant_identification": pid_role, "frozen_disease_model": frozen, "frozen_role": FROZEN_ROLE[frozen],
            "knowledge_base": kb_mode, "expected_report_type": expected,
            "disease_assessment_from_identification": False,
            "user_prompt": None if confirmed else "Please confirm the plant name; any identification shown is a suggestion only."}

