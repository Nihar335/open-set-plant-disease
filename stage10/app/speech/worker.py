"""Local speech worker (runs ONLY in the isolated speech environment C:/Users/nihar/venvs/agrovaidya-speech).

  <speech venv python> -m stage10.app.speech.worker      # started by stage10.app.speech.engines; JSON lines on stdin/stdout

Engines (SPEECH_DECISION.md, local/offline): Vosk (Apache-2.0) for speech-to-text, Kokoro-82M via kokoro-onnx (Apache-2.0 /
MIT; phonemiser espeak-ng GPL-3) for text-to-speech. Model files live in stage10_local/models/speech (hash-checked).

Privacy: networking is disabled inside this process before any engine is imported; audio arrives as base64 PCM on stdin,
is decoded in memory, and is released after each request; nothing is written to disk and no audio or text is logged.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import socket
import sys
from pathlib import Path


# ---------------------------------------------------------------------------------------------- no network, ever
class _NoNetwork(socket.socket):
    def connect(self, *a, **k):                                   # noqa: D401
        raise PermissionError("network access is disabled in the AgroVaidya speech worker")

    connect_ex = connect


socket.socket = _NoNetwork                                         # type: ignore[misc]
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(PermissionError("network disabled"))  # type: ignore
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "stage10_local" / "models" / "speech"
FILES = {  # sha256 recorded at download (2026-10-15)
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "kokoro-v1.0.int8.onnx": "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
VOSK = {"hi": os.environ.get("AGROVAIDYA_VOSK_HI", "vosk-model-small-hi-0.22"),
        "en": os.environ.get("AGROVAIDYA_VOSK_EN", "vosk-model-small-en-in-0.4")}
# runtime grammars (closed vocabularies) only work with small Vosk models
VOSK_GRAMMAR = {"en": os.environ.get("AGROVAIDYA_VOSK_EN_GRAMMAR", "vosk-model-small-en-in-0.4"),
                "hi": os.environ.get("AGROVAIDYA_VOSK_HI_GRAMMAR", "vosk-model-small-hi-0.22")}
VOICES = {"en": ("af_heart", "en-us"), "hi": ("hf_alpha", "hi")}
_state: dict = {}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _vosk(lang: str, grammar: bool = False):
    name = (VOSK_GRAMMAR if grammar else VOSK)[lang]
    key = "vosk_" + name
    if key not in _state:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        _state[key] = Model(str(MODELS / name))
    return _state[key]


def _kokoro():
    if "kokoro" not in _state:
        model = os.environ.get("AGROVAIDYA_TTS_MODEL", "kokoro-v1.0.int8.onnx")
        for f in (model, "voices-v1.0.bin"):
            if _sha(MODELS / f) != FILES[f]:
                raise RuntimeError(f"model file checksum mismatch: {f}")
        from kokoro_onnx import Kokoro
        _state["kokoro"] = Kokoro(str(MODELS / model), str(MODELS / "voices-v1.0.bin"))
        _state["kokoro_model"] = model
    return _state["kokoro"]


def transcribe(req: dict) -> dict:
    from vosk import KaldiRecognizer
    lang = req["language"]
    pcm = base64.b64decode(req["pcm16_b64"])
    grammar = req.get("grammar")
    model = _vosk(lang, grammar=bool(grammar))
    rec = KaldiRecognizer(model, 16000, json.dumps(grammar, ensure_ascii=False)) if grammar else KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    rec.SetMaxAlternatives(0)
    rec.AcceptWaveform(pcm)
    res = json.loads(rec.FinalResult())
    del pcm, rec
    words = res.get("result", [])
    conf = (sum(w["conf"] for w in words) / len(words)) if words else None
    return {"text": res.get("text", ""), "confidence": conf, "min_word_confidence": min((w["conf"] for w in words), default=None),
            "n_words": len(words), "engine": f"vosk/{(VOSK_GRAMMAR if grammar else VOSK)[lang]}"}


def synthesize(req: dict) -> dict:
    import numpy as np
    lang = req["language"]
    voice, kl = VOICES[lang]
    samples, sr = _kokoro().create(req["text"], voice=voice, speed=float(req.get("speed", 1.0)), lang=kl)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    out = {"pcm16_b64": base64.b64encode(pcm).decode("ascii"), "sample_rate": int(sr), "seconds": len(samples) / sr,
           "engine": f"kokoro/{_state['kokoro_model']}/{voice}"}
    del samples, pcm
    return out


def handle(msg: dict) -> dict:
    op = msg.get("op")
    if op == "ping":
        return {"ok": True, "models": {k: (MODELS / v).exists() for k, v in VOSK.items()} | {f: (MODELS / f).exists() for f in FILES}}
    if op == "load":                                               # warm-up: load a language model before the first request
        _vosk(msg["language"])
        if msg.get("tts"):
            _kokoro()
        return {"ok": True}
    if op == "transcribe":
        return {"ok": True, **transcribe(msg)}
    if op == "synthesize":
        return {"ok": True, **synthesize(msg)}
    if op == "network_probe":                                      # used by tests: proves the worker cannot reach out
        try:
            socket.create_connection(("example.com", 80), timeout=2)
            return {"ok": True, "network": "reachable"}
        except PermissionError:
            return {"ok": True, "network": "blocked"}
    return {"ok": False, "error": "unknown op"}


def main():
    stdin = sys.stdin
    out = sys.stdout
    sys.stdout = sys.stderr                                        # libraries may print; keep the protocol channel clean
    for line in stdin:
        msg: dict = {}
        try:
            msg = json.loads(line)
            resp = handle(msg)
        except Exception as e:                                     # noqa: BLE001 - only the type leaves the worker
            resp = {"ok": False, "error": type(e).__name__}
        finally:
            gc.collect()
        resp["id"] = msg.get("id") if isinstance(msg, dict) else None
        out.write(json.dumps(resp) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
