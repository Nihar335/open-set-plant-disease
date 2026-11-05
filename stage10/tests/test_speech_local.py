"""REAL local speech engines (Vosk speech-to-text, Kokoro text-to-speech) running in the isolated worker.

These tests use the installed engines, not fakes. They need the speech environment and models on this computer
(stage10/config.yaml speech.worker_python; stage10_local/models/speech). If they are missing the tests are SKIPPED with
that reason; they are never reported as passing without the engines. Speech used here is synthetic or public
(FLEURS); none of it is farmer speech, and nothing here is a usability result.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from stage10.app import server as SV
from stage10.app.speech import contract as SC
from stage10.app.speech import engines as E
from stage10.knowledge_base.kb import load_kb
from stage10.tests import report_fixtures as rf

ROOT = Path(__file__).resolve().parents[2]
STT, TTS, WHY = E.local_engines()
pytestmark = pytest.mark.skipif(STT is None, reason=f"local speech engines not installed: {WHY}")
KB = load_kb()


class FakeOSR:
    def assess(self, image):
        return rf.synthetic_assessment("Tomato___Early_blight", True)


@pytest.fixture(scope="module")
def client():
    c = SV.create_app(osr_factory=FakeOSR, kb=KB).test_client()
    yield c
    STT.worker.stop()


def to16k(wav: bytes) -> bytes:
    with wave.open(io.BytesIO(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)
        sr = w.getframerate()
    y = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x)
    return E.pcm16_to_wav(y.astype("<i2").tobytes(), 16000)


def say(c, text, lang):
    r = c.post("/api/v1/speech/synthesize", json={"text": text, "language": lang})
    assert r.status_code == 200 and r.mimetype == "audio/wav", r.get_data(as_text=True)[:200]
    return r.data


def hear(c, wav, lang, purpose=None):
    data = {"language": lang, "audio": (io.BytesIO(wav), "a.wav", "audio/wav")}
    if purpose:
        data["purpose"] = purpose
    return c.post("/api/v1/speech/transcribe", data=data, content_type="multipart/form-data")


def test_status_reports_real_local_engines(client):
    st = client.get("/api/v1/speech/status").json["data"]
    assert st["mode"] == "local" and st["speech_to_text"]["available"] and st["text_to_speech"]["available"]
    assert set(st["speech_to_text"]["languages"]) == {"en", "hi"} and st["privacy"]["audio_leaves_device"] is False
    assert "not verified" in st["language_status"]["en"] and "draft" in st["language_status"]["hi"]
    assert "microphone=(self)" in client.get("/").headers["Permissions-Policy"]


def test_worker_has_no_network_and_no_credentials():
    assert STT.worker.request({"op": "network_probe"})["network"] == "blocked"
    env = E.SpeechWorker._env()
    assert not [k for k in env if re.search(r"KEY|TOKEN|SECRET|PASS|PROXY|AUTH", k, re.I)]


def test_text_to_speech_english_and_hindi(client):
    for text, lang in (("Nothing in this report is a confirmed diagnosis.", "en"), ("इस रिपोर्ट में कोई भी बात पक्का निदान नहीं है।", "hi")):
        wav = say(client, text, lang)
        with wave.open(io.BytesIO(wav)) as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getnframes() / w.getframerate() > 1.0
    slow = say(client, "Nothing in this report is a confirmed diagnosis.", "en")
    r = client.post("/api/v1/speech/synthesize", json={"text": "x", "language": "en", "speed": 3})
    assert r.status_code == 422 and slow


def test_hindi_round_trip_through_the_api(client):
    text = "मेरे टमाटर के पत्तों पर भूरे धब्बे हैं"
    r = hear(client, to16k(say(client, text, "hi")), "hi")
    d = r.json["data"]
    assert r.status_code == 200 and d["status"] in ("ok", "low_confidence")
    assert d["requires_confirmation"] is True and d["confirmed"] is False
    hyp, ref = d["text"].split(), text.split()
    assert sum(w in hyp for w in ref) / len(ref) >= 0.7


def test_english_symptom_description_round_trip(client):
    r = hear(client, to16k(say(client, "The leaves have brown spots with rings and a yellow halo.", "en")), "en")
    d = r.json["data"]
    assert d["status"] in ("ok", "low_confidence") and "halo" in d["text"] and not d["confirmed"]


def test_plant_name_vocabulary_and_resolver(client):
    from stage10.workflow.plant_name import resolve_plant_name
    d = hear(client, to16k(say(client, "Potato", "en")), "en", purpose="plant_name").json["data"]
    assert d["text"] == "potato" and "[unk]" not in d["text"]
    assert resolve_plant_name(d["text"], KB)["plant_id"] == "PLANT-POTATO"


def test_silence_noise_long_and_invalid_audio_are_never_usable(client):
    silence = E.pcm16_to_wav(b"\0\0" * 16000, 16000)
    assert hear(client, silence, "en").json["data"]["status"] == "no_speech"
    rng = np.random.default_rng(0)
    noise = E.pcm16_to_wav((rng.normal(0, 2500, 32000)).astype("<i2").tobytes(), 16000)
    nd = hear(client, noise, "en")
    if nd.status_code == 200:
        assert nd.json["data"]["status"] in ("no_speech", "low_confidence", "ok") and nd.json["data"]["confirmed"] is False
    too_long = E.pcm16_to_wav(b"\x10\x00" * 16000 * 31, 16000)
    assert hear(client, too_long, "en").json["error"]["code"] == "speech_invalid_audio"
    assert hear(client, b"RIFF0000WAVEjunk", "en").json["error"]["code"] == "speech_invalid_audio"
    r = client.post("/api/v1/speech/transcribe", data={"language": "en", "audio": (io.BytesIO(b"\x1aE\xdf\xa3" * 100), "a.webm", "audio/webm")},
                    content_type="multipart/form-data")
    assert r.json["error"]["code"] == "speech_invalid_audio"                                  # local engines take WAV only
    assert hear(client, silence, "ta").json["error"]["code"] == "speech_unsupported_language"


def test_no_files_and_no_transcripts_logged_during_real_speech(client, caplog):
    dirs = [Path(tempfile.gettempdir()), ROOT / "stage10_local", ROOT / "stage10"]
    before = {d: {p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts} for d in dirs}
    secret = "my secret mango orchard has white powder"
    with caplog.at_level(logging.DEBUG):
        wav = say(client, secret, "en")
        d = hear(client, to16k(wav), "en").json
    after = {d_: {p for p in d_.rglob("*") if p.is_file() and "__pycache__" not in p.parts} for d_ in dirs}
    new = {str(p) for d_ in dirs for p in after[d_] - before[d_]}
    assert not [p for p in new if re.search(r"\.(wav|webm|ogg|pcm|raw|mp3)$", p, re.I)], new
    assert "orchard" not in caplog.text and "powder" not in caplog.text
    assert d["ok"] or d["error"]


def test_audio_helpers():
    st = np.stack([np.full(44100, 1000, "<i2"), np.full(44100, 3000, "<i2")], axis=1).reshape(-1).tobytes()
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100); w.writeframes(st)
    pcm, sec = E.wav_to_pcm16(b.getvalue())
    assert abs(sec - 1.0) < 1e-6 and abs(len(pcm) / 2 - 16000) <= 2
    assert abs(int(np.frombuffer(pcm, "<i2")[100]) - 2000) <= 1                              # mono = channel mean
    quiet = (np.full(1000, 200, "<i2")).tobytes()
    assert E.peak(E.normalise(quiet)) == min(23000, int(200 * E.MAX_GAIN))
    loud = (np.full(1000, 30000, "<i2")).tobytes()
    assert E.normalise(loud) == loud
    with pytest.raises(E.AudioError):
        E.wav_to_pcm16(b"not a wav")


def test_unknown_word_placeholders_are_removed():
    class F:
        name, version, processes_on_device = "f", "0", True

        def transcribe(self, *a):
            return SC.EngineResult("[unk] potato [unk]", 0.9)

    s = SC.SpeechSettings(mode="local", stt_languages=["en"], min_confidence=0.6, accepted_audio_types=["audio/wav"])
    assert SC.transcribe(b"RIFF" * 10, "audio/wav", "en", None, s, F())["text"] == "potato"


def test_model_files_match_recorded_hashes():
    from stage10.app.speech import worker as W
    for f, h in W.FILES.items():
        assert W._sha(W.MODELS / f) == h, f


def test_benchmark_is_recorded_and_honestly_labelled():
    b = json.loads((ROOT / "stage10/app/speech/benchmark_results.json").read_text(encoding="utf-8"))
    assert "NOT farmer speech" in b["note"] and b["stt"]["hi"]["n_utterances"] >= 100
    assert "SYNTHETIC" in b["plant_names_synthetic"]["data"]
