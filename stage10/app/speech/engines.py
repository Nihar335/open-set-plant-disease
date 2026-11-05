"""Local speech engines for the app (base environment side). The engines themselves run in the isolated speech worker
(worker.py, separate virtual environment, network disabled); this module only validates audio and talks to the worker
over stdin/stdout pipes. Nothing is written to disk; no audio or text is logged.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import queue
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np

from stage10.app.speech.contract import EngineResult
from stage10.schemas.validate import stage10_config

ROOT = Path(__file__).resolve().parents[3]
log = logging.getLogger("agrovaidya.speech")
SILENCE_PEAK = 60           # 16-bit peak below this (about -55 dBFS) = no speech; checked before any engine runs
MAX_GAIN = 40.0             # quiet microphones are normalised (peak -> about -3 dBFS), never amplified more than 40x
MAX_SECONDS = 30


class AudioError(ValueError):
    pass


def wav_to_pcm16(data: bytes) -> tuple[bytes, float]:
    """Mono 16-bit PCM at 16 kHz from a WAV upload (resampled if needed). Raises AudioError for anything else."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise AudioError("not a readable WAV recording") from e
    if width != 2 or ch not in (1, 2) or not 8000 <= sr <= 48000:
        raise AudioError("WAV must be 16-bit PCM, mono or stereo, 8-48 kHz")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    seconds = len(x) / sr
    if seconds > MAX_SECONDS:
        raise AudioError(f"recording longer than {MAX_SECONDS} seconds")
    if sr != 16000 and len(x):
        x = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype("<i2").tobytes(), seconds


def rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def peak(pcm: bytes) -> int:
    x = np.frombuffer(pcm, dtype="<i2")
    return int(np.abs(x.astype(np.int32)).max()) if len(x) else 0


def normalise(pcm: bytes) -> bytes:
    """Peak-normalise quiet recordings (in memory). Loud recordings are left unchanged."""
    p = peak(pcm)
    if p == 0:
        return pcm
    g = min(MAX_GAIN, 23000 / p)
    if g <= 1.0:
        return pcm
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) * g
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def pcm16_to_wav(pcm: bytes, sr: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return b.getvalue()


class SpeechWorker:
    """One long-lived worker process; requests are serialised. Restarted after a failure or timeout."""

    def __init__(self, python: str, timeout: float = 120.0):
        self.python, self.timeout = python, timeout
        self.proc = None
        self.lock = threading.Lock()
        self.n = 0

    def _start(self):
        self.proc = subprocess.Popen([self.python, "-m", "stage10.app.speech.worker"], cwd=str(ROOT), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                     env=self._env())

    @staticmethod
    def _env() -> dict:
        """Minimal environment: no credentials or proxy settings are passed to the worker (it has no network anyway)."""
        import os
        keep = ("SYSTEMROOT", "PATH", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "TEMP", "TMP", "LOCALAPPDATA", "NUMBER_OF_PROCESSORS")
        m = stage10_config().get("speech", {}).get("models", {})
        sel = {"AGROVAIDYA_VOSK_EN": m.get("stt_en"), "AGROVAIDYA_VOSK_HI": m.get("stt_hi"), "AGROVAIDYA_TTS_MODEL": m.get("tts"),
               "AGROVAIDYA_VOSK_EN_GRAMMAR": m.get("stt_en_names"), "AGROVAIDYA_VOSK_HI_GRAMMAR": m.get("stt_hi_names")}
        sel.update({k: os.environ[k] for k in sel if k in os.environ})          # explicit override (benchmarks)
        return {"PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1", **{k: os.environ[k] for k in keep if k in os.environ},
                **{k: v for k, v in sel.items() if v}}

    def request(self, msg: dict) -> dict:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.n += 1
            msg = {**msg, "id": self.n}
            q: queue.Queue = queue.Queue()
            try:
                self.proc.stdin.write(json.dumps(msg) + "\n")
                self.proc.stdin.flush()
                threading.Thread(target=lambda: q.put(self.proc.stdout.readline()), daemon=True).start()
                line = q.get(timeout=self.timeout)
            except (queue.Empty, OSError, BrokenPipeError) as e:
                self.stop()
                raise RuntimeError("speech worker did not respond") from e
            if not line:
                self.stop()
                raise RuntimeError("speech worker stopped")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise RuntimeError(f"speech worker error: {resp.get('error')}")
            return resp

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
            self.proc = None


class LocalSTT:
    name, processes_on_device = "vosk", True

    def __init__(self, worker: SpeechWorker):
        self.worker = worker
        self.version = "vosk-0.3.45"

    def transcribe(self, audio: bytes, mime: str, language: str, grammar: list | None = None) -> EngineResult:
        try:
            pcm, _ = wav_to_pcm16(audio)
        except AudioError:
            raise
        if peak(pcm) < SILENCE_PEAK:
            return EngineResult("", None)                            # silence: never sent to the recogniser
        pcm = normalise(pcm)
        r = self.worker.request({"op": "transcribe", "language": language, "pcm16_b64": base64.b64encode(pcm).decode("ascii"),
                                 "grammar": grammar})
        del pcm
        return EngineResult(r["text"], r["confidence"])


    def warm_up(self, language: str) -> None:
        self.worker.request({"op": "load", "language": language, "tts": True})


class LocalTTS:
    name, processes_on_device = "kokoro", True

    def __init__(self, worker: SpeechWorker):
        self.worker = worker
        self.version = "kokoro-onnx-0.6.1/kokoro-v1.0"

    def synthesize(self, text: str, language: str, speed: float = 1.0) -> bytes:
        r = self.worker.request({"op": "synthesize", "language": language, "text": text, "speed": speed})
        return pcm16_to_wav(base64.b64decode(r["pcm16_b64"]), r["sample_rate"])


def local_engines():
    """(stt, tts, reason) from config. Engines are None (with a reason) when the local set-up is incomplete."""
    c = stage10_config().get("speech", {})
    if c.get("mode") != "local":
        return None, None, "speech mode is not 'local'"
    py = Path(c.get("worker_python") or "")
    models = ROOT / "stage10_local" / "models" / "speech"
    if not py.exists():
        return None, None, "speech environment not installed"
    if not (models / "vosk-model-small-hi-0.22").exists() or not (models / "voices-v1.0.bin").exists():
        return None, None, "speech models not installed"
    w = SpeechWorker(str(py))
    return LocalSTT(w), LocalTTS(w), None
