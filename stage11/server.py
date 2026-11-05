"""Stage 11 web app: upload a leaf photo, type the plant name, pick the state -> automatic diagnosis.

Run:  python -m stage11.server        (http://127.0.0.1:8770, PORT env var overrides)
Photos are processed in memory and never stored or sent anywhere. Only the place name (optional) and the
coordinates of the place or state capital are sent to the Open-Meteo geocoding and weather APIs.

Public hosting: set LEAFSCAN_PUBLIC=1 (see deploy/huggingface). This turns on the request limits below and makes the
page state that photos go to an online server. HOST sets the listening address (default 127.0.0.1).
"""
from __future__ import annotations

import io
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from stage11.decision import THRESHOLD
from stage11.kb import load_kb

STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
app.json.sort_keys = False  # keep the report fields in the required order
_engine = None

PUBLIC = os.environ.get("LEAFSCAN_PUBLIC") == "1"
PER_VISITOR = (20, 600)    # at most 20 diagnoses per visitor in 10 minutes
ALL_VISITORS = (300, 3600)  # at most 300 diagnoses in an hour, across everyone
MAX_RUNNING = 2            # diagnoses running at once
MAX_WAIT_S = 20            # how long a request waits for a free slot
_slots = threading.BoundedSemaphore(MAX_RUNNING)
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _within(key: str, limit: int, window_s: int, now: float) -> bool:
    q = _hits[key]
    while q and q[0] <= now - window_s:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True


def _allowed() -> bool:
    """Basic protection against overload. The visitor address comes from the host's proxy header, so it is not proof of identity."""
    visitor = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
    now = time.monotonic()
    with _hits_lock:
        for key in [k for k, q in _hits.items() if k != "*" and (not q or q[-1] <= now - PER_VISITOR[1])]:
            del _hits[key]  # forget visitors whose window has passed, so this table cannot grow without limit
        return _within("ip:" + visitor, *PER_VISITOR, now) and _within("*", *ALL_VISITORS, now)


def engine():
    global _engine
    if _engine is None:
        from stage11.engine import Engine
        _engine = Engine()
    return _engine


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/api/meta")
def meta():
    kb = load_kb()
    supported = sorted({d["crop"] for d in kb.model_classes})
    return jsonify({
        "states": [s["name"] for s in kb.states],
        "plants": [{"name": p["common_name"], "scientific_name": p["scientific_name"],
                    "supported": kb.model_coverage(p["model_crop"])["level"] == "diagnosable",
                    "model_coverage": kb.model_coverage(p["model_crop"])["level"]}
                   for p in kb.plants],
        "supported_crops": supported,
        "threshold": THRESHOLD,
        "hosted": PUBLIC,
    })


@app.get("/api/diseases")
def diseases():
    """Disease library: every non-healthy knowledge-base entry, with whether the image model was trained on it."""
    kb = load_kb()
    sym = {s["id"]: s["name"] for s in kb.symptoms}
    names = {d["key"]: d["name"] for d in kb.diseases}
    out = []
    for d in kb.diseases:
        if d["pathogen_type"] == "none":
            continue
        out.append({
            "key": d["key"], "crop": d["crop"], "name": d["name"],
            "pathogen": d["pathogen"], "pathogen_type": d["pathogen_type"],
            "in_model": d["model_class"] is not None,
            "symptoms": [sym.get(k, k) for k, _ in sorted(d["symptoms"].items(), key=lambda kv: -kv[1])],
            "look_alikes": [names.get(k, k) for k in d["look_alikes"]],
            "weather": d["weather"],
            "tests": [kb.tests[t]["name"] for t in d["tests"] if t in kb.tests],
        })
    out.sort(key=lambda x: (x["crop"], x["name"]))
    return jsonify(out)


@app.get("/api/coverage")
def coverage():
    """What the image model can predict: its trained classes grouped by crop, with each crop's coverage level."""
    kb = load_kb()
    out = []
    for crop in dict.fromkeys(d["crop"] for d in kb.model_classes):
        cov = kb.model_coverage(crop)
        out.append({
            "crop": crop, "level": cov["level"],
            "plant_names": [p["common_name"] for p in kb.plants if p.get("model_crop") == crop],
            "diseases": [d["name"] for d in kb.model_classes if d["crop"] == crop and d["pathogen_type"] != "none"],
            "healthy": bool(cov["healthy"]),
        })
    return jsonify(out)


@app.get("/api/plant")
def plant_lookup():
    kb = load_kb()
    p, match = kb.resolve_plant(request.args.get("q", ""))
    if not p:
        return jsonify({"found": False})
    return jsonify({"found": True, "match": match, "common_name": p["common_name"],
                    "scientific_name": p["scientific_name"], "family": p["family"],
                    "supported": kb.model_coverage(p["model_crop"])["level"] == "diagnosable",
                    "model_coverage": kb.model_coverage(p["model_crop"])["level"]})


@app.post("/api/diagnose")
def diagnose():
    f = request.files.get("image")
    plant = (request.form.get("plant") or "").strip()
    state = (request.form.get("state") or "").strip()
    place = (request.form.get("place") or "").strip()[:100]
    if not f:
        return jsonify({"error": "Please upload a leaf photo."}), 400
    if not plant:
        return jsonify({"error": "Please enter the plant name."}), 400
    if not state:
        return jsonify({"error": "Please choose your state."}), 400
    data = f.read()
    try:
        from PIL import Image
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return jsonify({"error": "That file is not a readable image. Please send a JPEG, PNG or WebP photo "
                                 "(iPhone HEIC photos: choose 'Most Compatible' in camera settings, or share as JPEG)."}), 400
    if not load_kb().state(state):
        return jsonify({"error": f"Unknown state or union territory: {state}"}), 400
    if not PUBLIC:
        return jsonify(engine().diagnose(data, plant, state, place))
    if not _allowed():
        return jsonify({"error": "Too many requests right now. Please wait a few minutes and try again."}), 429
    if not _slots.acquire(timeout=MAX_WAIT_S):
        return jsonify({"error": "The app is busy right now. Please try again in a minute."}), 503
    try:
        return jsonify(engine().diagnose(data, plant, state, place))
    finally:
        _slots.release()


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Photo is larger than 15 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8770))
    host = os.environ.get("HOST", "127.0.0.1")
    engine()  # load the model before serving
    if PUBLIC:
        from waitress import serve
        serve(app, host=host, port=port, threads=4)
    else:
        app.run(host=host, port=port, debug=False, threaded=True)
