"""Voice input for the plant name: INTERFACE CONTRACT ONLY. Not operational.

Status (2026-10-15): no speech-to-text engine is installed, selected or approved. Engine choice is task M5-B2, blocked on
the owner's privacy decision (local vs external processing, consent). Nothing here downloads a model or calls a service.

Contract a future engine must satisfy (see stage10/workflow/VOICE_INPUT_CONTRACT.md):
  1. transcribe(audio) returns a Transcript (text, language, engine identity, no confidence shown as a percentage);
  2. the recognised text is SHOWN to the user, who confirms or edits it;
  3. only confirmed text enters PlantNameSession.enter(text, channel="voice_transcript_confirmed"), where it is resolved by
     exactly the same rules as typed text (no separate, looser matching for speech);
  4. audio is not stored or sent anywhere without the consent model of M5/M7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

VOICE_STATUS = {"engine": "local Vosk (offline); see stage10/app/speech/SPEECH_DECISION.md",
                "contract": "stage10/workflow/VOICE_INPUT_CONTRACT.md"}


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    engine_name: str
    engine_version: str
    processed_on_device: bool
    user_confirmed: bool = False


class SpeechToText(Protocol):
    name: str
    version: str

    def transcribe(self, audio: bytes, language: str) -> Transcript:
        """Recognised text for the user to confirm; never treated as confirmed by itself."""


def voice_input_available() -> bool:
    """True only when local speech is configured AND its engines are installed (stage10/app/speech/engines.py)."""
    from stage10.schemas.validate import stage10_config
    c = stage10_config().get("speech", {})
    if c.get("mode") != "local" or not c.get("stt_languages"):
        return False
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    return Path(c.get("worker_python") or "").exists() and (root / "stage10_local/models/speech").exists()
