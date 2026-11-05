"""Spoken-output script (M5-E2 / M5-E4): the text a future text-to-speech engine would read, built only from the
structured report. No speech engine exists (M5-E1 blocked on the owner's privacy decision); this module produces text.

Rules: every safety warning and the disclaimer are included in full; every candidate is introduced as "possible, not
confirmed"; no scores or percentages are spoken; whether the disease model ran is stated.
"""
from __future__ import annotations

TTS_STATUS = {"engine": "local Kokoro-82M (kokoro-onnx); see stage10/app/speech/SPEECH_DECISION.md"}
NOT_CONFIRMED = "possible, not confirmed"


def _t(obj, lang):
    if isinstance(obj, dict):
        return obj.get(lang) or obj.get("en") or ""
    return str(obj or "")


def _lang(obj, lang):
    """Language the text is actually in (Hindi drafts exist only for catalogue texts; knowledge-base text is English)."""
    return lang if isinstance(obj, dict) and obj.get(lang) else "en"


def script(report: dict, lang: str = "en") -> list[dict]:
    pi = report["plant_identity"]
    seg = []
    plant = pi["crop_confirmed_by_user"]
    seg.append({"kind": "plant", "text": f"Plant: {plant}, as you told us." if plant else "Plant: not known."})
    seg.append({"kind": "model", "text": "The disease photo model was used for this plant." if report["model_prediction"]["ran"]
                else "The disease photo model was not used for this report."})
    if report.get("insufficient_evidence_statement"):
        seg.append({"kind": "result", "text": _t(report["insufficient_evidence_statement"], lang),
                    "lang": _lang(report["insufficient_evidence_statement"], lang)})
    for c in report["candidate_diagnoses"]:
        seg.append({"kind": "candidate", "text": f"{_t(c['candidate_name'], 'en')}: {NOT_CONFIRMED}.", "lang": "en"})
    for st in report["uncertainty"]["statements"]:
        en = _t(st, "en")
        if en.startswith("Unanswered questions that would help"):
            seg.append({"kind": "question", "text": en, "lang": "en"})
    seg.append({"kind": "uncertainty", "text": f"Certainty of this report: {report['uncertainty']['level'].replace('_', ' ')}."})
    for group in ("immediate_actions",):
        for r in report[group]:
            seg.append({"kind": "action", "text": _t(r["text"], lang), "lang": _lang(r["text"], lang)})
    for w in report["safety_warnings"]:
        seg.append({"kind": "warning", "text": _t(w, lang), "lang": _lang(w, lang)})
    seg.append({"kind": "disclaimer", "text": _t(report["disclaimer"], lang), "lang": _lang(report["disclaimer"], lang)})
    for s_ in seg:
        s_.setdefault("lang", "en")
    return seg
