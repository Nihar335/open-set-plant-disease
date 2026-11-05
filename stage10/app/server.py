"""AgroVaidya farmer-facing application: local Flask API + static web app (Milestone 5).

  python -m stage10.app.server                      # http://127.0.0.1:8765  (AGROVAIDYA_HOST / AGROVAIDYA_PORT override)

Design rules:
  * all assessment logic is server-side and reuses the verified Stage-10 modules (plant-name workflow, routing, frozen
    disease-model adapter, knowledge base, report engine); the browser only collects input and renders the report;
  * automatic plant identification is never imported or called (scope change SC-M4-03);
  * uploaded photos are processed in memory and not written to disk; nothing is stored between requests;
  * no external service is contacted; the page loads no third-party assets;
  * errors are returned as {"ok": false, "error": {"code", "message"}} without stack traces or file paths.
Configuration (environment variables): AGROVAIDYA_HOST (default 127.0.0.1), AGROVAIDYA_PORT (8765),
AGROVAIDYA_DEVICE (cpu | cuda; default: the predictor's own choice).
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

from flask import Flask, Request, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from stage10.app import case_record, farmer_summary, marking, spoken, symptom_text
from stage10.app.speech import contract as SC
from stage10.knowledge_base.kb import load_kb
from stage10.report_engine.engine import ReportEngineError
from stage10.report_engine.inputs import INDIAN_STATES_AND_UTS, ImageInfo, Region, ReportInputError, USER_CHANNELS
from stage10.schemas.validate import stage10_config
from stage10.vision import quality as Q
from stage10.vision import symptoms as SY
from stage10.workflow import voice
from stage10.workflow.assessment import AssessmentError, assess_plant_health
from stage10.workflow.plant_name import PlantNameError, PlantNameSession, resolve_plant_name, supported_plant_options

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATIC = HERE / "static"
I18N = HERE / "i18n"
API = "/api/v1"
APP_VERSION = "0.5.0"
log = logging.getLogger("agrovaidya")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def ok(data, status=200):
    return jsonify({"ok": True, "data": data}), status


@lru_cache(maxsize=1)
def final_test_md5() -> frozenset:
    """MD5s of Stage 1-9 final-test images: an upload with the same bytes is refused (data guard)."""
    man = ROOT / stage10_config()["data_guards"]["forbidden_manifest"]
    with open(man, newline="", encoding="utf-8") as f:
        return frozenset(r["md5"].lower() for r in csv.DictReader(f) if r.get("md5"))


class _InMemoryImageOSR:
    """Gives FrozenOSRAdapter.assess() the uploaded photo as an in-memory PIL image (nothing written to disk)."""

    def __init__(self, adapter, pil_image):
        self.adapter, self.im = adapter, pil_image

    def assess(self, _path):
        return self.adapter.assess(self.im)


class InMemoryRequest(Request):
    """Uploads are kept in memory: Werkzeug's default spools files over 500 KB to a temporary file on disk."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def _quiet(fn, *a):
    try:
        fn(*a)
    except Exception:                                        # noqa: BLE001 - warm-up failure surfaces on the next real request
        log.warning("speech warm-up failed")


def create_app(osr_factory=None, kb=None, stt_engine=None, tts_engine=None, speech_settings=None) -> Flask:
    """osr_factory: callable returning an object with assess(PIL image) (default: the frozen FrozenOSRAdapter, loaded on
    first use). Tests inject a fake."""
    app = Flask(__name__, static_folder=None)
    app.request_class = InMemoryRequest
    cfg = stage10_config()
    app.config["MAX_CONTENT_LENGTH"] = (cfg["image_quality"]["max_upload_mb"] + 1) * 1024 * 1024
    app.config["KB"] = kb or load_kb()
    lock = threading.Lock()
    state = {"osr": None, "osr_error": None}

    def get_osr():
        with lock:
            if state["osr"] is None:
                try:
                    if osr_factory is not None:
                        state["osr"] = osr_factory()
                    else:
                        from stage10.osr.osr_adapter import FrozenOSRAdapter
                        state["osr"] = FrozenOSRAdapter(device=os.environ.get("AGROVAIDYA_DEVICE") or None)
                    state["osr_error"] = None
                except Exception as e:                       # noqa: BLE001 - reported safely, logged locally
                    log.exception("disease model failed to load")
                    state["osr_error"] = type(e).__name__
                    raise ApiError(503, "disease_model_unavailable",
                                   "The disease photo model could not be started on this computer. Please try again later.") from e
            return state["osr"]

    kbx = app.config["KB"]
    sp = speech_settings or SC.SpeechSettings.from_config()
    if stt_engine is None and tts_engine is None and speech_settings is None:
        from stage10.app.speech.engines import local_engines
        stt_engine, tts_engine, _why = local_engines()
    sp_status = lambda: SC.status(sp, stt_engine, tts_engine)              # noqa: E731

    # ------------------------------------------------------------------------------------------------- errors
    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        body = {"code": e.code, "message": e.message}
        if e.details is not None:
            body["details"] = e.details
        return jsonify({"ok": False, "error": body}), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_e):
        return jsonify({"ok": False, "error": {"code": "image_too_large", "message":
                        f"The photo is larger than {cfg['image_quality']['max_upload_mb']} MB. Please choose a smaller photo.",
                        "details": {"retake_requests": [{"issue": "too_large", **Q.RETAKE["too_large"]}]}}}), 413

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        return jsonify({"ok": False, "error": {"code": "http_" + str(e.code), "message": e.name}}), e.code

    @app.errorhandler(Exception)
    def _unexpected(e):
        log.exception("unexpected error")
        return jsonify({"ok": False, "error": {"code": "internal_error",
                                               "message": "Something went wrong on this computer. Nothing was saved. Please try again."}}), 500

    @app.after_request
    def _headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
                                                   "style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'self'")
        mic = "(self)" if sp_status()["speech_to_text"]["available"] else "()"
        resp.headers["Permissions-Policy"] = f"camera=(self), microphone={mic}, geolocation=()"
        if request.path.startswith(API):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ------------------------------------------------------------------------------------------------- read-only
    @app.get(f"{API}/health")
    def health():
        from stage10.osr.osr_adapter import verify_protected_files
        try:
            verify_protected_files()
            protected = True
        except Exception:                                    # noqa: BLE001
            protected = False
        return ok({"status": "ok", "app_version": APP_VERSION, "knowledge_base_version": kbx.version,
                   "disease_model": {"loaded": state["osr"] is not None, "protected_files_intact": protected,
                                     "last_load_error": state["osr_error"], "crops": cfg["frozen_model"]["supported_crops"]},
                   "plant_identification": case_record.PLANT_IDENTIFICATION,
                   "symptom_highlighting": case_record.SYMPTOM_EXTRACTION,
                   "voice_input": {**voice.VOICE_STATUS, "available": sp_status()["speech_to_text"]["available"]},
                   "voice_output": {**spoken.TTS_STATUS, "available": sp_status()["text_to_speech"]["available"]},
                   "ui_languages": cfg["language"].get("ui_languages", []),
                   "external_services": [], "images_stored": False})

    @app.get(f"{API}/plants")
    def plants():
        return ok({"plants": supported_plant_options(kbx),
                   "note": "Only these plants are in the knowledge base. Six of them can be checked by the disease photo model."})

    @app.post(f"{API}/plant-name/resolve")
    def resolve():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "text" not in body:
            raise ApiError(400, "bad_request", "Send JSON with a 'text' field.")
        return ok(resolve_plant_name(body["text"], kbx))

    @app.get(f"{API}/symptoms")
    def symptoms():
        feats = [{"feature_id": f, "label": x["label"], "category": x["category"],
                  "observable_from_single_photo": x.get("observable_from_single_photo")} for f, x in kbx.features.items()]
        ill = {i["question_id"]: {"src": "illustrations/" + i["file"], "alt": i["alt"], "caption": i["caption"]}
               for i in json.loads((STATIC / "illustrations" / "illustrations.json").read_text(encoding="utf-8"))["items"]}
        qs = [{"question_id": q, "text": x["text"], "options": [{"option_id": o["option_id"], "label": o["label"]} for o in x["options"]],
               "illustration": ill.get(q)} for q, x in kbx.questions.items()]
        return ok({"features": feats, "questions": qs, "knowledge_base_version": kbx.version})

    @app.post(f"{API}/symptoms/suggest")
    def suggest():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ApiError(400, "bad_request", "Send JSON with a 'text' field.")
        try:
            return ok(symptom_text.suggest(body.get("text"), kbx))
        except symptom_text.SymptomTextError as e:
            raise ApiError(422, "description_invalid", str(e).capitalize() + ".") from e

    @app.get(f"{API}/regions")
    def regions():
        return ok({"states": list(INDIAN_STATES_AND_UTS), "note": "State only. We never ask for your exact location."})

    @app.get(f"{API}/i18n/<lang>")
    def i18n(lang):
        codes = {x["code"] for x in cfg["language"].get("ui_languages", [])} | set(cfg["language"]["supported"])
        if lang not in codes or not (I18N / f"{lang}.json").exists():
            raise ApiError(404, "language_not_available", "This language is not available.")
        return ok(json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8")))

    # ------------------------------------------------------------------------------------------------- speech (contract)
    @app.get(f"{API}/speech/status")
    def speech_status():
        return ok(sp_status())

    @app.post(f"{API}/speech/warmup")
    def speech_warmup():
        lang = (request.get_json(silent=True) or {}).get("language")
        if not sp_status()["speech_to_text"]["available"]:
            raise ApiError(503, "speech_unavailable", "Voice is not available on this computer right now.")
        if lang not in sp.stt_languages:
            raise ApiError(422, "speech_unsupported_language", "Voice is not available in this language yet.")
        if hasattr(stt_engine, "warm_up"):
            threading.Thread(target=lambda: _quiet(stt_engine.warm_up, lang), daemon=True).start()
        return ok({"warming_up": lang})

    @app.post(f"{API}/speech/transcribe")
    def speech_transcribe():
        f = request.files.get("audio")
        audio = f.read() if f is not None else None
        grammar = None
        if request.form.get("purpose") == "plant_name" and request.form.get("language") == "en":
            # closed vocabulary for English plant names: the knowledge base's own names plus an "unknown" token
            grammar = SC.plant_name_grammar(kbx)
        res = SC.transcribe(audio, f.mimetype if f is not None else None, request.form.get("language"),
                            request.form.get("consent_version"), sp, stt_engine, grammar)
        del audio
        code = {"ok": 200, "low_confidence": 200, "no_speech": 200, "unavailable": 503, "service_error": 502,
                "consent_required": 403, "unsupported_language": 422, "invalid_audio": 422}[res["status"]]
        if code != 200:
            raise ApiError(code, "speech_" + res["status"], res["message"], {"speech": res})
        return ok(res)

    @app.post(f"{API}/speech/synthesize")
    def speech_synthesize():
        if not sp_status()["text_to_speech"]["available"]:
            raise ApiError(503, "speech_unavailable", "Reading aloud is not available yet.")
        body = request.get_json(silent=True) or {}
        err = SC.synthesis_request_errors(body.get("text"), body.get("language"), sp)
        if err:
            raise ApiError(422, "speech_request_invalid", "; ".join(err).capitalize() + ".")
        speed = body.get("speed", 1.0)
        if not isinstance(speed, (int, float)) or not 0.6 <= speed <= 1.4:
            raise ApiError(422, "speech_request_invalid", "Speed must be between 0.6 and 1.4.")
        try:
            wav = tts_engine.synthesize(body["text"], body["language"], speed) if speed != 1.0 else tts_engine.synthesize(body["text"], body["language"])
        except Exception as e:                                # noqa: BLE001
            log.warning("speech output engine failed (text not logged)")
            raise ApiError(502, "speech_service_error", "Reading aloud did not work this time.") from e
        from flask import Response
        return Response(wav, mimetype="audio/wav", headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------------------------------------- image check
    def _read_image():
        f = request.files.get("image")
        if f is None or not f.filename:
            raise ApiError(400, "image_missing", "Please add a photo of the plant.",
                           {"retake_requests": [{"issue": "missing", "view": "a photo of the affected leaf or plant",
                                                 "why": "A photo is needed for every check."}]})
        data = f.read()
        if not data:
            raise ApiError(400, "image_missing", "The photo file is empty. Please choose another photo.")
        res = Q.check_bytes(data)
        if res.md5 in final_test_md5():
            raise ApiError(403, "image_not_allowed", "This photo cannot be used here. Please take your own photo of the plant.")
        if any(i in res.issues for i in ("too_large", "bad_format", "unreadable")):
            raise ApiError(422, "image_invalid", "This file cannot be used: " + "; ".join(res.messages()) + ".",
                           {"retake_requests": res.retake_requests()})
        return data, res

    def _quality_json(res):
        return {"passed": res.passed, "issues": list(res.issues), "messages": res.messages(), "retake_requests": res.retake_requests(),
                "not_assessed": list(res.not_assessed), "width": res.width, "height": res.height, "format": res.format,
                "stored": False}

    @app.post(f"{API}/image/check")
    def image_check():
        _, res = _read_image()
        return ok(_quality_json(res))

    # ------------------------------------------------------------------------------------------------- assessment
    def _choice(raw):
        if not isinstance(raw, dict):
            raise ApiError(400, "plant_missing", "Please choose the plant first, or say that you do not know it.")
        s = PlantNameSession(kbx)
        action = raw.get("action")
        try:
            channel = raw.get("channel", "typed")
            if channel not in ("typed", "voice_transcript_confirmed"):
                raise ApiError(422, "plant_invalid", "Unknown entry channel.")
            if action == "typed":
                r = s.enter(raw.get("text"), channel=channel)
                if r["status"] != "resolved":
                    raise ApiError(422, "plant_unresolved", r["message"], {"resolution": r})
            elif action == "selected":
                s.select(raw.get("plant_id"))
            elif action == "unlisted":
                r = s.enter(raw.get("text"))
                if r["status"] not in ("unknown", "needs_clarification"):
                    raise ApiError(422, "plant_unresolved", r["message"], {"resolution": r})
                s.keep_unlisted_name()
            elif action == "dont_know":
                s.dont_know()
            else:
                raise ApiError(400, "plant_missing", "Please choose the plant first, or say that you do not know it.")
            if raw.get("confirmed") is not True:
                raise ApiError(422, "plant_not_confirmed", "Please confirm the plant name before continuing.")
            c = s.confirm()
        except PlantNameError as e:
            raise ApiError(422, "plant_invalid", str(e).capitalize() + ".") from e
        n = raw.get("corrections", 0)
        c["corrections"] = n if isinstance(n, int) and 0 <= n <= 50 else 0
        return c

    def _observations(form):
        try:
            feats = json.loads(form.get("features") or "{}")
            chans = json.loads(form.get("channels") or "{}")
            answers = json.loads(form.get("answers") or "{}")
        except json.JSONDecodeError as e:
            raise ApiError(400, "bad_request", "The symptom information could not be read.") from e
        if not all(isinstance(x, dict) for x in (feats, chans, answers)) or len(feats) > 60 or len(answers) > 30:
            raise ApiError(400, "bad_request", "The symptom information could not be read.")
        bad = [c for c in chans.values() if c not in USER_CHANNELS]
        if bad:
            raise ApiError(422, "observation_unconfirmed", "Only symptoms you have confirmed can be used.")
        return feats, chans, answers

    @app.post(f"{API}/assess")
    def assess():
        from PIL import Image
        form = request.form
        try:
            plant_raw = json.loads(form.get("plant") or "null")
        except json.JSONDecodeError as e:
            raise ApiError(400, "bad_request", "The plant choice could not be read.") from e
        choice = _choice(plant_raw)
        feats, chans, answers = _observations(form)
        state_name = form.get("state") or None
        lang = form.get("language") or "en"
        if lang not in cfg["language"]["supported"]:
            raise ApiError(422, "language_not_available", "This report language is not available.")
        data, res = _read_image()
        try:
            marks_raw = json.loads(form.get("marks") or "[]")
            mw, mh, mo = marking.oriented_size(data)
            marks = marking.validate_marks(marks_raw, mw, mh)
        except json.JSONDecodeError as e:
            raise ApiError(400, "bad_request", "The marked areas could not be read.") from e
        except marking.MarkError as e:
            raise ApiError(422, "marks_invalid", "The marked areas are not valid: " + str(e) + ".") from e
        info = ImageInfo(sha256=res.sha256, width=res.width, height=res.height, format=res.format if res.format in ("JPEG", "PNG") else "JPEG",
                         quality_issues=tuple(res.messages()), quality_not_assessed=res.not_assessed)
        from stage10.workflow.assessment import routing_decision
        try:
            dec = routing_decision(choice, kbx, info)
        except AssessmentError as e:
            raise ApiError(422, "plant_invalid", "The plant choice is not valid. Please choose the plant again.") from e
        adapter = None
        if dec["invoke_disease_model"]:
            adapter = _InMemoryImageOSR(get_osr(), Image.open(io.BytesIO(data)).convert("RGB"))
        # visible signs (colour/shape analysis, in memory); skipped when the photo failed the quality gate
        analysis = None if res.issues else SY.analyse(data)
        try:
            out = assess_plant_health(choice, None, kbx, reported_features=feats, reported_channels=chans, answers=answers,
                                      region=Region(state=state_name), language=lang, osr_adapter=adapter, image_info=info,
                                      image_observations=analysis.observations() if analysis and analysis.ok else ())
        except ReportInputError as e:
            raise ApiError(422, "input_invalid", str(e).capitalize() + ".") from e
        except (AssessmentError, ReportEngineError) as e:
            log.exception("assessment failed checks")
            raise ApiError(500, "assessment_failed", "The report could not be produced safely, so none is shown. Nothing was saved.") from e
        rep = out["report"]
        case = case_record.build_case(out["workflow"], rep, res, feats, chans, answers, kbx)
        case["stages"]["user_marked_regions"] = marking.marks_stage(marks, mw, mh, mo)
        summary = farmer_summary.build(kbx, rep, out["workflow"], analysis.to_dict() if analysis else None)
        return ok({"workflow": out["workflow"], "report": rep, "case": case, "summary": summary,
                   "image_quality": _quality_json(res),
                   "spoken_script": {"available": sp_status()["text_to_speech"]["available"], "status": spoken.TTS_STATUS,
                                     "segments": spoken.script(rep, lang)}})

    # ------------------------------------------------------------------------------------------------- static app
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/<path:path>")
    def static_files(path):
        if path.startswith("api/"):
            raise ApiError(404, "not_found", "Unknown address.")
        if not (STATIC / path).resolve().is_relative_to(STATIC.resolve()) or not (STATIC / path).is_file():
            return send_from_directory(STATIC, "index.html")
        return send_from_directory(STATIC, path)

    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("AGROVAIDYA_HOST", "127.0.0.1")
    port = int(os.environ.get("AGROVAIDYA_PORT") or os.environ.get("PORT") or "8765")     # PORT: set by preview tools
    app = create_app()
    print(f"AgroVaidya running at http://{host}:{port}  (local only; press Ctrl+C to stop)")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
