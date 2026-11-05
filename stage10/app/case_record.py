"""Case record (M5-A4): every stage's output is stored separately, and a stage that did not run says so.

Stages: plant_name (user) | image_quality | plant_identification (disabled) | symptom_extraction (not available) |
user_observations | disease_model | report. A failed or skipped stage never produces output for another stage.
Nothing is persisted: the record lives only in the response to the request that created it (M7 governs storage).
"""
from __future__ import annotations

SYMPTOM_EXTRACTION = {"status": "not_available",
                      "reason": "No validated method exists to find symptoms in photos (task M5-A5). Nothing is highlighted "
                                "or inferred from the image."}
PLANT_IDENTIFICATION = {"status": "disabled",
                        "reason": "Automatic plant identification is not used (scope change SC-M4-03). The plant name is yours."}


def build_case(workflow: dict, report: dict, quality, features: dict, channels: dict, answers: dict, kb) -> dict:
    sel = workflow["plant_selection"]
    ran = report["model_prediction"]["ran"]
    obs = [{"feature_id": f, "label": kb.features[f]["label"], "reported": "present" if v else "absent",
            "input_channel": channels.get(f, "symptom_checklist"), "status": "confirmed_by_user"} for f, v in sorted(features.items())]
    ans = [{"question_id": q, "option_id": o, "question": kb.questions[q]["text"],
            "answer": next(x["label"] for x in kb.questions[q]["options"] if x["option_id"] == o)} for q, o in sorted(answers.items())]
    return {
        "case_id": report["report_id"],
        "stages": {
            "plant_name": {"status": "confirmed_by_user" if sel["outcome"] != "not_known" else "not_known",
                           "outcome": sel["outcome"], "plant_name": sel["plant_name"], "plant_id": sel["plant_id"],
                           "entered_text": sel["entered_text"], "source": "user_supplied", "independently_verified": False},
            "image_quality": {"status": "passed" if quality.passed else "failed", "issues": list(quality.issues),
                              "not_assessed": list(quality.not_assessed), "retake_requests": quality.retake_requests(),
                              "width": quality.width, "height": quality.height, "format": quality.format},
            "plant_identification": dict(PLANT_IDENTIFICATION),
            "symptom_extraction": dict(SYMPTOM_EXTRACTION),
            "user_observations": {"status": "provided" if obs or ans else "none", "observations": obs, "answers": ans},
            "disease_model": {"status": "ran" if ran else "not_run", "reason": workflow["routing"]["reason"],
                              "decision": report["osr_status"]["decision"] if ran else None,
                              "predicted_class": report["model_prediction"]["predicted_class"] if ran else None},
            "report": {"status": "generated", "report_id": report["report_id"], "report_type": report["report_type"]},
        },
        "stored": False,
    }
