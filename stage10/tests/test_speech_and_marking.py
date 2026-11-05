"""M5 speech contract (no engine installed) and farmer-guided symptom marking.

Speech: fake engines only; nothing is downloaded. Marking: synthetic photos; marks never reach the assessment.
"""
from __future__ import annotations

import copy
import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stage10.app import marking as MK
from stage10.app import server as SV
from stage10.app.speech import build_evidence as EVD
from stage10.app.speech import contract as SC
from stage10.knowledge_base.kb import load_kb
from stage10.schemas.validate import stage10_config
from stage10.tests import report_fixtures as rf
from stage10.workflow import plant_name as PN

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "stage10" / "app"
KB = load_kb()
SECRET = "kaddu ke patte peele"                      # a transcript that must never appear in logs


class FakeSTT:
    name, version, processes_on_device = "fake-stt", "0", True

    def __init__(self, text=SECRET, confidence=0.9, fail=False):
        self.text, self.confidence, self.fail, self.calls = text, confidence, fail, 0

    def transcribe(self, audio, mime, language):
        self.calls += 1
        if self.fail:
            raise RuntimeError("engine exploded at C:/models/secret")
        return SC.EngineResult(self.text, self.confidence, ("alt one", ""))


class FakeTTS:
    name, version, processes_on_device = "fake-tts", "0", True

    def __init__(self, fail=False):
        self.fail = fail

    def synthesize(self, text, language):
        if self.fail:
            raise RuntimeError("boom")
        return b"RIFF....WAVEfake"


class FakeOSR:
    def assess(self, image):
        return rf.synthetic_assessment("Tomato___Early_blight", True)


LOCAL = dict(mode="local", stt_languages=["hi", "en"], tts_languages=["hi"], min_confidence=0.5)
AUDIO = b"\x1aE\xdf\xa3" + b"\x00" * 2000


def settings(**kw):
    return SC.SpeechSettings(**{**LOCAL, **kw})


# ------------------------------------------------------------------------------------------ speech: contract
def test_config_is_local_offline_and_private():
    c = stage10_config()["speech"]
    assert c["mode"] == "local" and c["external_provider"] is None and c["retain_audio"] is False
    assert c["accepted_audio_types"] == ["audio/wav"] and set(c["stt_languages"]) <= {"en", "hi"}
    for lang in c["stt_languages"] + c["tts_languages"]:                 # never claimed verified without evidence
        assert "not verified" in c["language_status"][lang] or "draft" in c["language_status"][lang]


def test_every_transcript_needs_confirmation():
    r = SC.transcribe(AUDIO, "audio/webm;codecs=opus", "hi", None, settings(), FakeSTT())
    assert r["status"] == "ok" and r["text"] == SECRET and r["requires_confirmation"] and not r["confirmed"]
    assert r["alternatives"] == ["alt one"] and r["audio_retained"] is False


@pytest.mark.parametrize("kw,expect", [
    (dict(engine=FakeSTT(confidence=0.2)), "low_confidence"),
    (dict(engine=FakeSTT(confidence=None)), "low_confidence"),
    (dict(engine=FakeSTT(text="   ")), "no_speech"),
    (dict(language="ta"), "unsupported_language"),
    (dict(audio=b""), "invalid_audio"),
    (dict(audio=b"x" * 2_000_001), "invalid_audio"),
    (dict(mime="video/mp4"), "invalid_audio"),
    (dict(engine=FakeSTT(fail=True)), "service_error"),
    (dict(engine=None), "unavailable"),
    (dict(settings=SC.SpeechSettings()), "unavailable"),
    (dict(settings=settings(mode="external", external_provider="acme")), "consent_required"),
    (dict(settings=settings(mode="external", external_provider=None), consent="speech-consent-v1"), "consent_required"),
])
def test_failure_statuses_never_yield_a_usable_result(kw, expect):
    args = dict(audio=AUDIO, mime="audio/webm", language="hi", consent=None, settings=settings(), engine=FakeSTT())
    args.update(kw)
    r = SC.transcribe(args["audio"], args["mime"], args["language"], args["consent"], args["settings"], args["engine"])
    assert r["status"] == expect and r["requires_confirmation"] and not r["confirmed"]
    assert r["must_repeat_or_type"] is True
    if expect != "low_confidence":
        assert r["text"] is None
    assert "secret" not in json.dumps(r) and "exploded" not in json.dumps(r)


def test_external_processing_only_with_provider_and_consent():
    s = settings(mode="external", external_provider="approved-provider")
    assert SC.transcribe(AUDIO, "audio/wav", "hi", "speech-consent-v0", s, FakeSTT())["status"] == "consent_required"
    assert SC.transcribe(AUDIO, "audio/wav", "hi", "speech-consent-v1", s, FakeSTT())["status"] == "ok"
    st = SC.status(s, FakeSTT(), None)
    assert st["privacy"]["audio_leaves_device"] is True and st["consent"]["external_processing"] is True


# ------------------------------------------------------------------------------------------ speech: API
def test_disabled_speech_reports_unavailable_and_blocks_the_microphone():
    c = SV.create_app(osr_factory=FakeOSR, kb=KB, speech_settings=SC.SpeechSettings()).test_client()
    st = c.get("/api/v1/speech/status").json["data"]
    assert st["speech_to_text"]["available"] is False and st["text_to_speech"]["available"] is False
    assert st["privacy"] == {"audio_retained": False, "audio_logged": False, "transcripts_logged": False, "audio_leaves_device": False}
    r = c.post("/api/v1/speech/transcribe", data={"language": "hi", "audio": (io.BytesIO(AUDIO), "a.webm", "audio/webm")},
               content_type="multipart/form-data")
    assert r.status_code == 503 and r.json["error"]["code"] == "speech_unavailable"
    assert c.post("/api/v1/speech/synthesize", json={"text": "x", "language": "hi"}).status_code == 503
    assert "microphone=()" in c.get("/").headers["Permissions-Policy"]
    h = c.get("/api/v1/health").json["data"]
    assert h["voice_input"]["available"] is False and h["voice_output"]["available"] is False


def test_speech_api_with_a_fake_engine_and_no_logging(caplog, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("audio written to a temporary file")
    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", boom)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    stt = FakeSTT()
    c = SV.create_app(osr_factory=FakeOSR, kb=KB, stt_engine=stt, tts_engine=FakeTTS(), speech_settings=settings()).test_client()
    assert "microphone=(self)" in c.get("/").headers["Permissions-Policy"]
    with caplog.at_level(logging.DEBUG):
        r = c.post("/api/v1/speech/transcribe", data={"language": "hi", "audio": (io.BytesIO(AUDIO), "a.webm", "audio/webm")},
                   content_type="multipart/form-data")
        assert r.status_code == 200 and r.json["data"]["text"] == SECRET and r.json["data"]["confirmed"] is False
        bad = SV.create_app(osr_factory=FakeOSR, kb=KB, stt_engine=FakeSTT(fail=True), speech_settings=settings()).test_client()
        r2 = bad.post("/api/v1/speech/transcribe", data={"language": "hi", "audio": (io.BytesIO(AUDIO), "a.webm", "audio/webm")},
                      content_type="multipart/form-data")
        assert r2.status_code == 502 and "secret" not in r2.get_data(as_text=True)
        r3 = c.post("/api/v1/speech/transcribe", data={"language": "ta", "audio": (io.BytesIO(AUDIO), "a.webm", "audio/webm")},
                    content_type="multipart/form-data")
        assert r3.status_code == 422 and r3.json["error"]["code"] == "speech_unsupported_language"
        t = c.post("/api/v1/speech/synthesize", json={"text": "Namaste", "language": "hi"})
        assert t.status_code == 200 and t.mimetype == "audio/wav" and t.headers["Cache-Control"] == "no-store"
        assert c.post("/api/v1/speech/synthesize", json={"text": "x" * 2001, "language": "hi"}).status_code == 422
        assert c.post("/api/v1/speech/synthesize", json={"text": "x", "language": "ta"}).status_code == 422
    logs = caplog.text
    assert SECRET not in logs and "Namaste" not in logs and "C:/models" not in logs


def test_spoken_plant_names_follow_the_typed_rules(monkeypatch):
    from stage10.workflow import voice as V
    monkeypatch.setattr(V, "voice_input_available", lambda: False)
    with pytest.raises(PN.PlantNameError):
        PN.PlantNameSession(KB).enter("tomato", channel=PN.VOICE_CHANNEL)
    monkeypatch.setattr(V, "voice_input_available", lambda: True)
    assert PN.PlantNameSession(KB).enter("potato", channel=PN.VOICE_CHANNEL)["plant_id"] == "PLANT-POTATO"
    src = (ROOT / "stage10/workflow/plant_name.py").read_text(encoding="utf-8")
    assert "fuzzy" not in src.lower() or "never" in src.lower()


def test_no_credentials_or_provider_calls_in_the_frontend():
    js = "".join(p.read_text(encoding="utf-8") for p in (APP / "static").rglob("*.js"))
    assert not re.search(r"api[_-]?key|authorization|bearer|secret|x-api-key", js, re.I)
    assert not re.search(r"webkitSpeechRecognition\s*\(|new\s+SpeechRecognition|getUserMedia\(\{\s*audio:\s*true", js)
    assert "audio: false" in js                                          # the camera never opens the microphone


def test_speech_evidence_register_is_current_and_cited():
    doc, err = EVD.build()
    assert err == [] and len(doc["sources"]) == 31
    assert (Path(EVD.OUT)).read_text(encoding="utf-8") == json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    dec = (APP / "speech/SPEECH_DECISION.md").read_text(encoding="utf-8")
    cited = set(re.findall(r"\bE(\d{1,2})\b", dec))
    assert {str(i) for i in range(1, 32)} <= cited
    assert "IMPLEMENTED LOCALLY" in dec and "NOT YET VERIFIED" in dec and "NOT farmer speech" in dec   # no verification claimed


# ------------------------------------------------------------------------------------------ marking
def jpeg(size=(640, 480), orientation=None) -> bytes:
    rng = np.random.default_rng(1)
    im = Image.fromarray(rng.integers(40, 200, (size[1] // 8 + 1, size[0] // 8 + 1, 3), dtype=np.uint8)).resize(size, Image.NEAREST)
    b = io.BytesIO()
    if orientation:
        ex = im.getexif()
        ex[0x0112] = orientation
        im.save(b, "JPEG", exif=ex.tobytes())
    else:
        im.save(b, "JPEG")
    return b.getvalue()


def test_marks_are_scaled_to_the_displayed_image():
    out = MK.validate_marks([{"shape": "circle", "x": 0.5, "y": 0.25, "r": 0.1},
                             {"shape": "box", "x0": 0.8, "y0": 0.9, "x1": 0.2, "y1": 0.1, "via": "grid"}], 640, 480)
    assert out[0]["pixels"] == {"x": 320, "y": 120, "r": 48}
    assert out[1]["normalized"] == {"x0": 0.2, "y0": 0.1, "x1": 0.8, "y1": 0.9} and out[1]["pixels"] == {"x0": 128, "y0": 48, "x1": 512, "y1": 432}
    assert all(m["source"] == "user_marked" and m["verified"] is False and "not checked" in m["label"] for m in out)
    assert MK.validate_marks([], 640, 480) == [] and MK.validate_marks(None, 640, 480) == []


def test_rotated_photo_uses_displayed_orientation():
    assert MK.oriented_size(jpeg((640, 480), orientation=6)) == (480, 640, 6)
    assert MK.oriented_size(jpeg((640, 480))) == (640, 480, 1)
    w, h, _ = MK.oriented_size(jpeg((640, 480), orientation=6))
    assert MK.validate_marks([{"shape": "circle", "x": 0.5, "y": 0.75, "r": 0.1}], w, h)[0]["pixels"] == {"x": 240, "y": 480, "r": 48}


@pytest.mark.parametrize("bad", [
    "nope", [{"shape": "star"}], [{"shape": "circle", "x": 1.5, "y": 0.2, "r": 0.1}], [{"shape": "circle", "x": True, "y": 0.2, "r": 0.1}],
    [{"shape": "circle", "x": float("nan"), "y": 0.2, "r": 0.1}], [{"shape": "circle", "x": 0.5, "y": 0.5, "r": 0}],
    [{"shape": "box", "x0": 0.5, "y0": 0.5, "x1": 0.505, "y1": 0.9}], [{"shape": "box", "x0": 0.1, "y0": 0.1, "x1": 0.5}],
    [{"shape": "circle", "x": 0.5, "y": 0.5, "r": 0.1, "via": "ai"}], [{"shape": "circle", "x": 0.5, "y": 0.5, "r": 0.1}] * 11])
def test_invalid_marks_are_rejected(bad):
    with pytest.raises(MK.MarkError):
        MK.validate_marks(bad, 640, 480)


def _assess(c, marks=None, img=None):
    data = {"plant": json.dumps({"action": "typed", "text": "tomato", "confirmed": True}),
            "features": json.dumps({"FEAT-CONCENTRIC-RINGS": True}), "image": (io.BytesIO(img or jpeg()), "a.jpg")}
    if marks is not None:
        data["marks"] = marks if isinstance(marks, str) else json.dumps(marks)
    return c.post("/api/v1/assess", data=data, content_type="multipart/form-data")


def test_marks_are_recorded_but_never_change_the_assessment():
    c = SV.create_app(osr_factory=FakeOSR, kb=KB).test_client()
    plain = _assess(c).json["data"]
    marked = _assess(c, [{"shape": "circle", "x": 0.4, "y": 0.6, "r": 0.06}, {"shape": "box", "x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.4}]).json["data"]
    # identical report apart from the generation time and the report id derived from it (requests ~1 s apart)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("created_at", "report_id")}            # noqa: E731
    assert json.dumps(strip(plain["report"]), sort_keys=True) == json.dumps(strip(marked["report"]), sort_keys=True)
    assert set(plain["report"]) == set(marked["report"])
    st = marked["case"]["stages"]["user_marked_regions"]
    assert st["status"] == "marked_by_user" and st["used_in_assessment"] is False and len(st["regions"]) == 2
    assert st["coordinate_basis"]["width"] == 640 and "not available" in st["automatic_localisation"]
    assert plain["case"]["stages"]["user_marked_regions"]["status"] == "none"
    # anything listed as found in the photo comes from the automated analyser of the whole photo, never from the marks
    assert plain["report"]["visible_symptoms"] == marked["report"]["visible_symptoms"]
    assert all("colour_shape_v" in v["visual_evidence"]["en"] for v in marked["report"]["visible_symptoms"] if v["origin"] == "image_analysis")


def test_mark_errors_and_removal_via_api():
    c = SV.create_app(osr_factory=FakeOSR, kb=KB).test_client()
    assert _assess(c, "{oops").status_code == 400
    r = _assess(c, [{"shape": "circle", "x": 2, "y": 0.5, "r": 0.1}])
    assert r.status_code == 422 and r.json["error"]["code"] == "marks_invalid"
    assert _assess(c, []).json["data"]["case"]["stages"]["user_marked_regions"]["regions"] == []     # all marks removed
    junk = c.post("/api/v1/assess", data={"plant": json.dumps({"action": "typed", "text": "tomato", "confirmed": True}),
                                          "marks": json.dumps([]), "image": (io.BytesIO(b"not an image"), "a.jpg")},
                  content_type="multipart/form-data")
    assert junk.status_code == 422 and junk.json["error"]["code"] == "image_invalid"


def test_marking_and_voice_node_tests_pass():
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for the frontend unit tests")
    p = subprocess.run([node, "--test", str(APP / "tests_js" / "marking_voice.test.js")], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "# fail 0" in p.stdout, p.stdout[-2000:]


def test_ui_strings_for_marking_and_voice_exist():
    en = json.loads((APP / "i18n/en.json").read_text(encoding="utf-8"))["strings"]
    from stage10.app.speech.contract import STATUSES
    for st in STATUSES:
        if st != "ok":
            assert f"speech.{st}" in en, st
    js = (APP / "static/js/marking.js").read_text(encoding="utf-8")
    grid = re.search(r"GRID = \[(.*?)\]", js).group(1)
    for g in re.findall(r'"([a-z-]+)"', grid):
        assert f"mark.grid.{g}" in en
    assert "not checked by AgroVaidya" in en["mark.label"]


def test_original_photo_is_preserved_and_marks_stay_out_of_prediction(monkeypatch):
    seen = []

    class SpyOSR:
        def assess(self, image):
            seen.append(np.asarray(image).copy())                       # what the disease model actually receives
            return rf.synthetic_assessment("Tomato___Early_blight", True)

    c = SV.create_app(osr_factory=SpyOSR, kb=KB).test_client()
    img = jpeg()
    import hashlib
    a = _assess(c, img=img).json["data"]
    b = _assess(c, [{"shape": "box", "x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9}], img=img).json["data"]
    assert a["report"]["image_metadata"]["sha256"] == b["report"]["image_metadata"]["sha256"] == hashlib.sha256(img).hexdigest()
    assert len(seen) == 2 and np.array_equal(seen[0], seen[1])          # the model sees the unmarked original pixels
    assert b["case"]["stored"] is False and b["image_quality"]["stored"] is False
    src = (APP / "marking.py").read_text(encoding="utf-8")
    assert "save(" not in src and "write" not in src                    # marking never writes or alters the image


def test_corrected_transcript_replaces_the_heard_text_in_the_case_record():
    c = SV.create_app(osr_factory=FakeOSR, kb=KB).test_client()

    def plant(p):
        d = {"plant": json.dumps({"confirmed": True, **p}), "image": (io.BytesIO(jpeg()), "a.jpg")}
        return c.post("/api/v1/assess", data=d, content_type="multipart/form-data")

    heard = plant({"action": "typed", "text": "potato", "channel": "voice_transcript_confirmed"}).json["data"]
    assert heard["workflow"]["plant_selection"]["entry_channel"] == "voice_transcript_confirmed"
    # the user corrected "potatoes" -> "tomato" before confirming: the frontend sends the corrected text as typed
    fixed = plant({"action": "typed", "text": "tomato"}).json["data"]
    sel = fixed["workflow"]["plant_selection"]
    assert sel["entry_channel"] == "typed" and sel["entered_text"] == "tomato" and fixed["case"]["stages"]["plant_name"]["entered_text"] == "tomato"
    assert plant({"action": "typed", "text": "tomato", "channel": "voice_raw"}).status_code == 422          # no unconfirmed channel
