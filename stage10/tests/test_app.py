"""Milestone 5 application: Flask API, quality gate, case record, typed symptoms, spoken script, local aliases, i18n,
frontend unit tests (Node) and one end-to-end run through the REAL frozen disease model on a validation image.

Most tests use a FAKE disease-model adapter (synthetic assessments). No final-test image is opened.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from stage10.app import server as SV
from stage10.app import spoken, symptom_text
from stage10.knowledge_base.kb import load_kb
from stage10.schemas.validate import stage10_config
from stage10.tests import report_fixtures as rf
from stage10.vision import quality as Q
from stage10.workflow import plant_name as PN

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "stage10" / "app"
KB = load_kb()


class FakeOSR:
    calls = 0

    def __init__(self, cls="Tomato___Early_blight", known=True):
        self.cls, self.known = cls, known

    def assess(self, image):
        FakeOSR.calls += 1
        assert isinstance(image, Image.Image)          # in-memory photo, never a file path
        return rf.synthetic_assessment(self.cls, self.known)


@pytest.fixture()
def client():
    FakeOSR.calls = 0
    return SV.create_app(osr_factory=FakeOSR, kb=KB).test_client()


def photo(size=(640, 480), fmt="JPEG", blur=0, brightness=1.0, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (size[1] // 8 + 1, size[0] // 8 + 1, 3), dtype=np.uint8)
    im = Image.fromarray(base).resize(size, Image.NEAREST)          # sharp edges, mid exposure
    arr = np.asarray(im).astype(np.float32) * brightness
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def assess(c, plant, img=None, features=None, channels=None, answers=None, **form):
    data = {"plant": json.dumps(plant), "features": json.dumps(features or {}), "channels": json.dumps(channels or {}),
            "answers": json.dumps(answers or {}), **form}
    if img is not False:
        data["image"] = (io.BytesIO(img or photo()), "leaf.jpg")
    return c.post("/api/v1/assess", data=data, content_type="multipart/form-data")


P = lambda **kw: {"confirmed": True, **kw}                                  # noqa: E731
EB = {"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True}


# ------------------------------------------------------------------------------------------ read-only endpoints
def test_health_reports_capabilities_honestly(client):
    d = client.get("/api/v1/health").json["data"]
    assert d["plant_identification"]["status"] == "disabled"
    assert d["symptom_highlighting"]["status"] == "not_available"
    from stage10.workflow.voice import voice_input_available
    # voice availability must reflect the actually installed local engines (never claimed otherwise)
    assert d["voice_input"]["available"] is voice_input_available() and d["voice_output"]["available"] is voice_input_available()
    assert d["external_services"] == [] and d["images_stored"] is False
    assert d["disease_model"]["protected_files_intact"] is True


def test_plants_and_resolution_use_the_verified_resolver(client):
    plants = client.get("/api/v1/plants").json["data"]["plants"]
    assert {p["plant_id"] for p in plants} == set(KB.plants)
    assert {p["plant_id"] for p in plants if p["disease_model_supported"]} == {"PLANT-TOMATO", "PLANT-APPLE", "PLANT-GRAPE", "PLANT-MAIZE",
                                                                                "PLANT-PEPPER-BELL", "PLANT-POTATO"}
    cases = {"Paddy": "resolved", "  CORN ": "resolved", "": "empty", "tomatoe": "needs_clarification", "Coconut": "unknown", "x<1>": "invalid"}
    for text, st in cases.items():
        r = client.post("/api/v1/plant-name/resolve", json={"text": text}).json["data"]
        assert r["status"] == st, text
        assert r == PN.resolve_plant_name(text, KB)           # the API returns exactly the M4 resolver's output
    assert client.post("/api/v1/plant-name/resolve", json={"nope": 1}).status_code == 400


def test_no_frontend_name_table():
    js = "".join(p.read_text(encoding="utf-8") for p in (APP / "static" / "js").glob("*.js"))
    for alias in ("paddy", "Lycopersicon", "Zea mays", "grapevine", "PLANT-RICE"):
        assert alias not in js, alias                         # names come only from the server's resolver


def test_symptoms_and_regions(client):
    d = client.get("/api/v1/symptoms").json["data"]
    assert {f["feature_id"] for f in d["features"]} == set(KB.features)
    assert all(any(o["option_id"] in ("not_sure", "not_checked") for o in q["options"]) for q in d["questions"])
    assert len(client.get("/api/v1/regions").json["data"]["states"]) == 36


# ------------------------------------------------------------------------------------------ images
def test_image_check_valid_invalid_missing_and_too_large(client):
    ok = client.post("/api/v1/image/check", data={"image": (io.BytesIO(photo()), "a.jpg")}, content_type="multipart/form-data")
    assert ok.status_code == 200 and ok.json["data"]["passed"] is True and ok.json["data"]["stored"] is False
    gif = io.BytesIO()
    Image.new("RGB", (300, 300)).save(gif, "GIF")
    bad = client.post("/api/v1/image/check", data={"image": (io.BytesIO(gif.getvalue()), "a.gif")}, content_type="multipart/form-data")
    assert bad.status_code == 422 and bad.json["error"]["code"] == "image_invalid"
    junk = client.post("/api/v1/image/check", data={"image": (io.BytesIO(b"not an image"), "a.jpg")}, content_type="multipart/form-data")
    assert junk.status_code == 422 and junk.json["data"] if junk.json.get("data") else junk.json["error"]["code"] == "image_invalid"
    miss = client.post("/api/v1/image/check", data={}, content_type="multipart/form-data")
    assert miss.status_code == 400 and miss.json["error"]["code"] == "image_missing"
    big = b"\xff\xd8" + os.urandom(12 * 1024 * 1024)
    r = client.post("/api/v1/image/check", data={"image": (io.BytesIO(big), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 413 and r.json["error"]["code"] == "image_too_large" and "Traceback" not in r.get_data(as_text=True)


@pytest.mark.parametrize("kw,issue", [({"size": (180, 150)}, "too_small"), ({"blur": 6}, "blurry"),
                                      ({"brightness": 0.06}, "too_dark")])
def test_quality_gate_flags_synthetic_degradations_with_retake_requests(client, kw, issue):
    r = client.post("/api/v1/image/check", data={"image": (io.BytesIO(photo(**kw)), "a.jpg")}, content_type="multipart/form-data").json["data"]
    assert not r["passed"] and issue in r["issues"]
    req = next(x for x in r["retake_requests"] if x["issue"] == issue)
    assert req["view"] and req["why"]                           # names the needed view and why (M5-A3)


def test_overexposure_and_every_issue_has_a_retake_request():
    im = Image.fromarray(np.full((480, 640, 3), 250, np.uint8))
    b = io.BytesIO()
    im.save(b, "PNG")
    assert "too_bright" in Q.check_bytes(b.getvalue()).issues
    assert set(Q.RETAKE) == {"too_large", "bad_format", "unreadable", "too_small", "blurry", "too_dark", "too_bright"}


def test_quality_thresholds_come_from_the_validation_calibration():
    cal = json.loads((ROOT / "stage10/vision/quality_calibration.json").read_text(encoding="utf-8"))
    cfg = stage10_config()["image_quality"]
    assert cfg["blur_laplacian_var_min"] == cal["blur_laplacian_var_min"] and cfg["brightness_range"] == cal["brightness_range"]
    assert cal["data"].startswith("Stage 1-9 known-class validation") and cal["fraction_flagged"]["orig"]["blurry"] <= 0.01
    assert cal["fraction_flagged"]["blur4"]["blurry"] >= 0.95
    src = (ROOT / "stage10/vision/calibrate_quality.py").read_text(encoding="utf-8")
    assert "val.csv" in src and "test.csv" not in src and "assert_not_final_test" in src


def test_final_test_images_are_refused_by_content(client, monkeypatch):
    with open(ROOT / "data/splits/test.csv", newline="", encoding="utf-8") as f:
        assert next(csv.DictReader(f))["md5"].lower() in SV.final_test_md5()      # real guard set covers the test split
    data = photo(seed=7)
    import hashlib
    monkeypatch.setattr(SV, "final_test_md5", lambda: frozenset({hashlib.md5(data).hexdigest()}))
    r = client.post("/api/v1/image/check", data={"image": (io.BytesIO(data), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 403 and r.json["error"]["code"] == "image_not_allowed"


def test_uploads_never_touch_the_disk(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("upload spooled to a temporary file")
    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", boom)
    monkeypatch.setattr(tempfile, "TemporaryFile", boom)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    b = io.BytesIO()
    Image.effect_noise((1200, 900), 64).convert("RGB").save(b, "PNG")
    big = b.getvalue()                                           # > 500 KB, Werkzeug's default spool threshold
    assert len(big) > 600_000
    r = assess(client, P(action="typed", text="tomato"), img=big)
    assert r.status_code == 200, r.json


# ------------------------------------------------------------------------------------------ assessment routing
def test_supported_crop_runs_the_disease_model_once(client):
    r = assess(client, P(action="typed", text="Tomato"), features=EB).json["data"]
    assert FakeOSR.calls == 1 and r["report"]["report_type"] == "known_disease" and r["report"]["model_prediction"]["ran"]
    assert r["case"]["stages"]["disease_model"]["status"] == "ran"
    assert r["report"]["plant_identity"]["crop_confirmed_by_user"] == "Tomato"


@pytest.mark.parametrize("plant,rtype", [(P(action="typed", text="mango"), "out_of_model_scope"),
                                         (P(action="selected", plant_id="PLANT-RICE"), "out_of_model_scope"),
                                         (P(action="unlisted", text="Coconut"), "out_of_model_scope"),
                                         (P(action="dont_know"), "unknown_investigation")])
def test_unsupported_unknown_and_unlisted_never_invoke_the_model(client, plant, rtype):
    r = assess(client, plant, features={"FEAT-FRUIT-LESIONS": True, "FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": True}).json["data"]
    rep = r["report"]
    assert FakeOSR.calls == 0 and rep["model_prediction"]["ran"] is False and rep["report_type"] == rtype
    assert not any(c["ranking_method"] == "model_class_evidence" for c in rep["candidate_diagnoses"])
    assert r["case"]["stages"]["disease_model"]["status"] == "not_run"


def test_unknown_plant_never_gets_a_fabricated_diagnosis(client):
    rep = assess(client, P(action="unlisted", text="Coconut"), features=EB).json["data"]["report"]
    assert rep["candidate_diagnoses"] == [] and rep["insufficient_evidence_statement"]


def test_quality_failure_blocks_the_model_and_requests_a_retake(client):
    r = assess(client, P(action="typed", text="tomato"), img=photo(blur=6)).json["data"]
    assert FakeOSR.calls == 0 and r["report"]["report_type"] == "insufficient_image"
    st = r["case"]["stages"]
    assert st["image_quality"]["status"] == "failed" and st["disease_model"]["status"] == "not_run"
    assert st["image_quality"]["retake_requests"]


@pytest.mark.parametrize("plant,code", [(None, "plant_missing"), ({"action": "typed", "text": "tomato"}, "plant_not_confirmed"),
                                        (P(action="typed", text="tomatoe"), "plant_unresolved"),
                                        (P(action="unlisted", text="tomato"), "plant_unresolved"),
                                        (P(action="selected", plant_id="PLANT-COCONUT"), "plant_invalid"),
                                        (P(action="typed", text=""), "plant_unresolved")])
def test_plant_input_errors(client, plant, code):
    r = assess(client, plant)
    assert r.status_code in (400, 422) and r.json["error"]["code"] == code and FakeOSR.calls == 0


def test_correction_count_is_recorded(client):
    r = assess(client, P(action="selected", plant_id="PLANT-POTATO", corrections=2)).json["data"]
    assert r["workflow"]["plant_selection"]["corrections"] == 2


def test_missing_image_and_bad_observations(client):
    r = assess(client, P(action="typed", text="tomato"), img=False)
    assert r.status_code == 400 and r.json["error"]["code"] == "image_missing"
    r = assess(client, P(action="typed", text="tomato"), features={"FEAT-NOPE": True})
    assert r.status_code == 422 and r.json["error"]["code"] == "input_invalid"
    r = assess(client, P(action="typed", text="tomato"), features={"FEAT-CONCENTRIC-RINGS": True},
               channels={"FEAT-CONCENTRIC-RINGS": "voice_transcript"})
    assert r.status_code == 422 and r.json["error"]["code"] == "observation_unconfirmed"
    r = client.post("/api/v1/assess", data={"plant": "{not json"}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_user_observations_stay_distinct_from_model_and_knowledge_base(client):
    r = assess(client, P(action="typed", text="tomato"), features={"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": False},
               channels={"FEAT-CONCENTRIC-RINGS": "typed_text_confirmed"}, answers={"Q-LEAF-AGE": "older_lower"}).json["data"]
    vs = r["report"]["visible_symptoms"]
    user = [s for s in vs if s["origin"] == "user_report"]
    assert user and all(s["input_channel"] in ("symptom_checklist", "typed_text_confirmed", "guided_question_answer") for s in user)
    assert all(s["evidence_status"] != "observed" for s in vs if s["origin"] == "knowledge_base_typical")
    # image-found signs (SC-M5-02 colour/shape analyser) are always labelled as automated, inferred and low-confidence
    for s_ in (s for s in vs if s["origin"] == "image_analysis"):
        assert s_["evidence_status"] == "inferred" and s_["confidence"] == "low" and s_["input_channel"] == "image_analysis"
        assert "colour_shape_v" in s_["visual_evidence"]["en"] and "unvalidated" in s_["visual_evidence"]["en"]
    obs = r["case"]["stages"]["user_observations"]
    assert {o["feature_id"]: o["reported"] for o in obs["observations"]} == {"FEAT-CONCENTRIC-RINGS": "present", "FEAT-YELLOW-HALO": "absent"}
    assert obs["answers"][0]["question_id"] == "Q-LEAF-AGE"
    assert r["case"]["stages"]["symptom_extraction"]["status"] == "not_available"
    assert r["case"]["stages"]["plant_identification"]["status"] == "disabled" and r["case"]["stored"] is False


def test_hindi_report_is_marked_as_draft(client):
    r = assess(client, P(action="typed", text="tomato"), language="hi").json["data"]["report"]
    assert r["language"] == "hi" and r["review_status"]["translation_review"]["hi"] == "pending_fluent_speaker_review"
    assert assess(client, P(action="typed", text="tomato"), language="fr").status_code == 422


def test_model_load_failure_and_internal_errors_are_safe(monkeypatch):
    def broken():
        raise RuntimeError("C:/secret/path/model.pt missing")
    c = SV.create_app(osr_factory=broken, kb=KB).test_client()
    r = assess(c, P(action="typed", text="tomato"))
    body = r.get_data(as_text=True)
    assert r.status_code == 503 and r.json["error"]["code"] == "disease_model_unavailable" and "secret" not in body
    assert assess(c, P(action="typed", text="mango")).status_code == 200      # other routes keep working
    monkeypatch.setattr(SV, "assess_plant_health", lambda *a, **k: 1 / 0)
    c2 = SV.create_app(osr_factory=FakeOSR, kb=KB).test_client()
    r = assess(c2, P(action="typed", text="mango"))
    assert r.status_code == 500 and r.json["error"]["code"] == "internal_error" and "ZeroDivision" not in r.get_data(as_text=True)


def test_security_headers_static_files_and_unknown_api(client):
    r = client.get("/")
    assert r.status_code == 200 and b'<div id="app"' in r.data
    csp = r.headers["Content-Security-Policy"]
    pp = r.headers["Permissions-Policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "connect-src 'self'" in csp
    # the microphone is limited to this local site, and only when local speech is available (else fully blocked)
    from stage10.workflow.voice import voice_input_available
    assert ("microphone=(self)" if voice_input_available() else "microphone=()") in pp and "microphone=*" not in pp
    assert client.get("/api/v1/nope").status_code == 404
    assert client.get("/../config.yaml").status_code in (200, 404) and b"frozen_model" not in client.get("/../config.yaml").data
    for f in (APP / "static").rglob("*"):
        if f.suffix in (".html", ".css", ".js"):
            t = f.read_text(encoding="utf-8")
            assert not re.search(r"https?://(?!www\.w3\.org/2000/svg|127\.0\.0\.1|localhost)", t), f   # nothing from the internet


def test_identification_code_is_never_loaded_by_the_app():
    code = ("import sys, io, json; from stage10.app import server as SV; from stage10.tests import report_fixtures as rf\n"
            "class F:\n    def assess(self, i): return rf.synthetic_assessment('Tomato___Early_blight', True)\n"
            "c = SV.create_app(osr_factory=F).test_client()\n"
            "from PIL import Image; b = io.BytesIO(); Image.effect_noise((640, 480), 60).convert('RGB').save(b, 'JPEG')\n"
            "r = c.post('/api/v1/assess', data={'plant': json.dumps({'action': 'typed', 'text': 'tomato', 'confirmed': True}),"
            " 'image': (io.BytesIO(b.getvalue()), 'a.jpg')}, content_type='multipart/form-data'); assert r.status_code == 200, r.json\n"
            "bad = [m for m in ('stage10.plant_id.adapter', 'stage10.plant_id.pilot.evaluate_pilot', 'open_clip') if m in sys.modules]\n"
            "print(bad); sys.exit(1 if bad else 0)")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


# ------------------------------------------------------------------------------------------ typed symptoms, spoken script
def test_typed_description_gives_suggestions_only(client):
    r = client.post("/api/v1/symptoms/suggest", json={"text": "Brown spots with rings and a yellow halo, zzqx"}).json["data"]
    ids = [s["feature_id"] for s in r["suggestions"]]
    assert "FEAT-CONCENTRIC-RINGS" in ids and "FEAT-YELLOW-HALO" in ids
    assert all(s["status"] == "suggested_needs_confirmation" for s in r["suggestions"]) and "zzqx" in r["not_understood"]
    assert r["text_stored"] is False
    for bad in ("", "x" * 501, "<script>alert(1)</script>"):
        assert client.post("/api/v1/symptoms/suggest", json={"text": bad}).status_code == 422
    assert all(f in KB.features for fs in symptom_text.CUES.values() for f in fs)


def _all_types(client):
    out = {"known_disease": assess(client, P(action="typed", text="tomato"), features=EB),
           "unknown_investigation": assess(client, P(action="dont_know"), features=EB),
           "out_of_model_scope": assess(client, P(action="typed", text="mango"), features={"FEAT-FRUIT-LESIONS": True,
                                                                                           "FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": True}),
           "insufficient_image": assess(client, P(action="typed", text="tomato"), img=photo(size=(150, 150)))}
    return {k: v.json["data"] for k, v in out.items()}


def test_spoken_script_keeps_every_warning_and_not_confirmed_label_for_every_report_type(client):
    for rtype, d in _all_types(client).items():
        assert d["report"]["report_type"] == rtype
        seg = d["spoken_script"]["segments"]
        text = " ".join(s["text"] for s in seg)
        for w in d["report"]["safety_warnings"]:
            assert w["en"] in text, rtype
        assert d["report"]["disclaimer"]["en"] in text
        for c in d["report"]["candidate_diagnoses"]:
            assert f"{c['candidate_name']['en']}: {spoken.NOT_CONFIRMED}" in text
        assert not re.search(r"\d\s*%|\d\.\d{2,}", text)          # no percentages or numeric scores are spoken
        assert d["spoken_script"]["available"] is client.get("/api/v1/speech/status").json["data"]["text_to_speech"]["available"]
        assert all(s["lang"] in ("en", "hi") for s in seg)


# ------------------------------------------------------------------------------------------ localisation
def test_ui_catalogue_covers_every_key_used_and_hindi_is_not_claimed():
    en = json.loads((APP / "i18n/en.json").read_text(encoding="utf-8"))
    hi = json.loads((APP / "i18n/hi.json").read_text(encoding="utf-8"))
    js = "".join(p.read_text(encoding="utf-8") for p in (APP / "static/js").glob("*.js"))
    used = {k for k in re.findall(r'\bt\("([a-z0-9_.]+)"', js) if not k.endswith((".", "_"))}   # prefix + value keys: see `dynamic`
    dynamic = {f"steps.{s}" for s in ("plant", "photo", "symptoms", "review", "report")} | \
              {f"symptoms.cat.{c}" for c in ("lesion", "colour_pattern", "sign", "distribution", "damage", "plant_part", "environment")} | \
              {f"report.banner.{r}" for r in ("known_disease", "unknown_investigation", "out_of_model_scope", "insufficient_image")} | \
              {f"report.uncertainty.{u}" for u in ("low", "moderate", "high", "not_assessable")} | \
              {f"help.how.{i}" for i in range(1, 5)} | {f"help.privacy.{i}" for i in range(1, 6)} | \
              {f"help.limits.{i}" for i in range(1, 7)} | {"help.languages.1", "home.limits.1", "home.limits.2", "home.limits.3"} | \
              {"voice.status_en", "voice.status_hi"} | {f"help.voice.{i}" for i in range(1, 5)} | \
              {f"player.kind.{k}" for k in ("plant", "model", "result", "candidate", "uncertainty", "action", "warning", "disclaimer",
                                            "question")} | {f"voice.consent.{i}" for i in range(1, 5)} | \
              {f"speech.{x}" for x in ("unavailable", "cancelled", "type_instead", "denied", "no_device", "low_confidence", "no_speech",
                                       "unsupported_language", "invalid_audio", "consent_required", "service_error")}
    missing = sorted((used | dynamic) - set(en["strings"]))
    assert missing == []
    assert hi["meta"]["review_status"] == "machine_draft_unreviewed"          # SC-M5-02 draft: never claimed as reviewed
    assert "not yet checked" in en["strings"]["symptoms.language.hi"]


def test_local_alias_mechanism_requires_review_and_never_guesses(tmp_path):
    assert PN.reviewed_local_aliases(KB) == []                               # none reviewed yet: none invented
    f = tmp_path / "a.yaml"
    f.write_text("aliases:\n- {alias: Aam, lang: hi, plant_id: PLANT-MANGO, editorial: true, review_status: draft}\n", encoding="utf-8")
    assert PN.reviewed_local_aliases(KB, f) == []                            # drafts are never used
    f.write_text("aliases:\n- {alias: Aam, lang: hi, plant_id: PLANT-MANGO, editorial: true, review_status: reviewed}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        PN.reviewed_local_aliases(KB, f)                                     # reviewed without reviewer/date is malformed
    f.write_text("aliases:\n- {alias: Aam, lang: hi, plant_id: PLANT-MANGO, review_status: reviewed, reviewer: X, reviewed_on: 2026-01-01}\n",
                 encoding="utf-8")
    with pytest.raises(ValueError):
        PN.reviewed_local_aliases(KB, f)                                     # neither source nor editorial
    f.write_text("aliases:\n- {alias: Aam, lang: hi, plant_id: PLANT-MANGO, editorial: true, review_status: reviewed, reviewer: X, "
                 "reviewed_on: 2026-01-01}\n- {alias: Aam, lang: hi, plant_id: PLANT-POTATO, editorial: true, review_status: reviewed, "
                 "reviewer: X, reviewed_on: 2026-01-01}\n", encoding="utf-8")
    entries = PN.reviewed_local_aliases(KB, f)
    assert len(entries) == 2


# ------------------------------------------------------------------------------------------ frontend unit tests (Node)
def test_frontend_unit_tests_pass(client, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for the frontend unit tests (node --test)")
    for rtype, d in _all_types(client).items():
        (tmp_path / f"{rtype}.json").write_text(json.dumps(d), encoding="utf-8")
    p = subprocess.run([node, "--test", str(APP / "tests_js" / "frontend.test.js")], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "AGROVAIDYA_FIXTURES": str(tmp_path)}, timeout=120)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    assert "# fail 0" in p.stdout and "# skipped 0" in p.stdout


# ------------------------------------------------------------------------------------------ real end-to-end
def test_end_to_end_with_the_real_frozen_disease_model():
    """Plant name -> photo (Stage 1-9 VALIDATION image, never test) -> real FrozenOSRAdapter -> report."""
    with open(ROOT / "data/splits/val.csv", newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["class_name"] == "Tomato___Early_blight" and r["split"] == "val")
    from stage10.guards import assert_not_final_test
    assert_not_final_test(ROOT / row["image_path"])
    c = SV.create_app(kb=KB).test_client()                                  # default factory: the real frozen adapter
    img = (ROOT / row["image_path"]).read_bytes()
    r = assess(c, P(action="typed", text="tomato"), img=img, features={"FEAT-CONCENTRIC-RINGS": True})
    assert r.status_code == 200, r.json
    d = r.json["data"]
    rep = d["report"]
    assert rep["model_prediction"]["ran"] is True and rep["model_prediction"]["model_versions"]
    assert rep["report_type"] in ("known_disease", "unknown_investigation")
    assert rep["plant_identity"]["crop_confirmed_by_user"] == "Tomato" and d["workflow"]["plant_selection"]["independently_verified"] is False
    assert c.get("/api/v1/health").json["data"]["disease_model"]["loaded"] is True
    r2 = assess(c, P(action="typed", text="mango"), img=img)
    assert r2.json["data"]["report"]["model_prediction"]["ran"] is False
