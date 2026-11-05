"""Speech interface contracts (M5-B2..B5, M5-E1..E4). NO ENGINE IS INSTALLED: with the configured mode "disabled" every
request returns status "unavailable". See SPEECH_DECISION.md for the recommended engines and the owner decisions.

Contract (binding for any future engine):
  * a transcript is ALWAYS returned unconfirmed (requires_confirmation=True); the user must confirm, correct, repeat or
    cancel it, and only confirmed text may enter the plant-name resolver or symptom suggestions (same rules as typing);
  * low confidence, no speech, unsupported language, invalid audio and engine failure each have their own status and
    never produce a usable plant name or symptom;
  * audio is processed in memory only and never written to disk or logs; transcripts are never logged;
  * external processing requires an approved provider AND the user's consent for this session (consent_version);
  * provider credentials, if ever needed, are read from server environment variables only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from stage10.schemas.validate import stage10_config

log = logging.getLogger("agrovaidya.speech")
STATUSES = ("ok", "low_confidence", "no_speech", "unsupported_language", "invalid_audio", "consent_required", "unavailable",
            "service_error")
USER_MESSAGES = {
    "ok": "Is this what you said? Please confirm, correct it, or say it again.",
    "low_confidence": "We are not sure we heard you correctly. Please say it again, or type it.",
    "no_speech": "We did not hear anything. Please try again closer to the phone, or type instead.",
    "unsupported_language": "Voice is not available in this language yet. Please type or tap instead.",
    "invalid_audio": "The recording could not be used. Please try again.",
    "consent_required": "Please read and accept how your voice will be processed before using voice input.",
    "unavailable": "Voice input is not available yet. Please type or tap instead.",
    "service_error": "Voice recognition did not work this time. Please try again, or type instead.",
}


@dataclass(frozen=True)
class EngineResult:
    text: str
    confidence: float | None = None          # engine's own score; never shown as a percentage
    alternatives: tuple = ()


class SpeechToTextEngine(Protocol):
    name: str
    version: str
    processes_on_device: bool

    def transcribe(self, audio: bytes, mime: str, language: str) -> EngineResult: ...


class TextToSpeechEngine(Protocol):
    name: str
    version: str
    processes_on_device: bool

    def synthesize(self, text: str, language: str) -> bytes: ...


@dataclass
class SpeechSettings:
    mode: str = "disabled"
    stt_languages: list = field(default_factory=list)
    tts_languages: list = field(default_factory=list)
    max_audio_bytes: int = 2_000_000
    accepted_audio_types: list = field(default_factory=lambda: ["audio/webm", "audio/ogg", "audio/wav"])
    min_confidence: float | None = None
    max_tts_chars: int = 2000
    consent_version: str = "speech-consent-v1"
    external_provider: str | None = None
    language_status: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> "SpeechSettings":
        c = stage10_config().get("speech", {})
        return cls(**{k: c[k] for k in cls.__dataclass_fields__ if k in c})


def plant_name_grammar(kb) -> list[str]:
    """Closed vocabulary for spoken English plant names: the knowledge base's common names and plain synonyms only
    (scientific names and bracketed forms are left out: farmers do not say them and their parts mislead the decoder)."""
    import re
    names = set()
    for p in kb.plants.values():
        common = [x["name"] for x in p["common_names"]]
        sci = {p["scientific_name"].lower(), p["taxonomy"]["species"].lower()}
        for raw in common + p.get("synonyms", []):
            n = raw.lower().strip()
            latin = raw not in common and re.fullmatch(r"[A-Z][a-z]+ [a-z]+", raw.strip()) is not None   # e.g. Lycopersicon esculentum
            if n.isascii() and "(" not in n and n not in sci and not latin:
                names.add(n)
    return sorted(names) + ["[unk]"]


def status(settings: SpeechSettings, stt=None, tts=None) -> dict:
    stt_ok = settings.mode != "disabled" and stt is not None and bool(settings.stt_languages)
    tts_ok = settings.mode != "disabled" and tts is not None and bool(settings.tts_languages)
    return {"mode": settings.mode,
            "speech_to_text": {"available": stt_ok, "languages": list(settings.stt_languages) if stt_ok else [],
                               "engine": getattr(stt, "name", None) if stt_ok else None},
            "text_to_speech": {"available": tts_ok, "languages": list(settings.tts_languages) if tts_ok else [],
                               "engine": getattr(tts, "name", None) if tts_ok else None},
            "language_status": {k: v for k, v in settings.language_status.items() if stt_ok or tts_ok},
            "consent": {"required": True, "version": settings.consent_version,
                        "external_processing": settings.mode == "external", "provider": settings.external_provider},
            "privacy": {"audio_retained": False, "audio_logged": False, "transcripts_logged": False,
                        "audio_leaves_device": settings.mode == "external"},
            "decision_record": "stage10/app/speech/SPEECH_DECISION.md"}


def _result(st: str, text: str | None = None, alternatives=(), engine=None) -> dict:
    assert st in STATUSES
    usable = st in ("ok", "low_confidence")
    return {"status": st, "text": text if usable else None, "alternatives": list(alternatives) if usable else [],
            "requires_confirmation": True, "confirmed": False, "message": USER_MESSAGES[st],
            "must_repeat_or_type": st != "ok", "engine": engine, "audio_retained": False}


def transcribe(audio: bytes | None, mime: str | None, language: str | None, consent_version: str | None,
               settings: SpeechSettings, engine: SpeechToTextEngine | None, grammar: list | None = None) -> dict:
    """Validate and run one transcription. Never returns a confirmed result; never logs audio or text."""
    if settings.mode == "disabled" or engine is None:
        return _result("unavailable")
    if settings.mode == "external" and (consent_version != settings.consent_version or not settings.external_provider):
        return _result("consent_required")
    if language not in settings.stt_languages:
        return _result("unsupported_language")
    base_mime = (mime or "").split(";")[0].strip().lower()
    if not audio or base_mime not in settings.accepted_audio_types or len(audio) > settings.max_audio_bytes:
        return _result("invalid_audio")
    try:
        r = engine.transcribe(audio, base_mime, language, grammar) if grammar else engine.transcribe(audio, base_mime, language)
    except ValueError:                                         # unreadable / too long / wrong format (engines.AudioError)
        return _result("invalid_audio", engine=engine.name)
    except Exception:                                          # noqa: BLE001 - engine internals never reach the user
        log.warning("speech engine failed (details withheld: no audio or transcript is logged)")
        return _result("service_error", engine=engine.name)
    finally:
        del audio                                              # no reference kept; nothing written anywhere
    text = " ".join(w for w in (r.text or "").split() if w != "[unk]").strip()   # recogniser placeholder for unknown words
    if not text:
        return _result("no_speech", engine=engine.name)
    alts = [a for a in r.alternatives if isinstance(a, str) and a.strip()][:3]
    if settings.min_confidence is not None and (r.confidence is None or r.confidence < settings.min_confidence):
        return _result("low_confidence", text, alts, engine.name)
    return _result("ok", text, alts, engine.name)


def synthesis_request_errors(text: str | None, language: str | None, settings: SpeechSettings) -> list[str]:
    err = []
    if not isinstance(text, str) or not text.strip():
        err.append("empty text")
    elif len(text) > settings.max_tts_chars:
        err.append("text too long")
    if language not in settings.tts_languages:
        err.append("language not available for speech output")
    return err
