"""Stage 11 web app: upload a leaf photo, type the plant name, pick the state -> automatic diagnosis.

Run:  python -m stage11.server        (http://127.0.0.1:8770, PORT env var overrides)
Photos are processed in memory and never stored or sent anywhere. Only the place name (optional) and the
coordinates of the place or state capital are sent to the Open-Meteo geocoding and weather APIs.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from stage11.decision import THRESHOLD
from stage11.kb import load_kb

STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
app.json.sort_keys = False  # keep the report fields in the required order
_engine = None


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
    return jsonify(engine().diagnose(data, plant, state, place))


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Photo is larger than 15 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8770))
    engine()  # load the model before serving
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
