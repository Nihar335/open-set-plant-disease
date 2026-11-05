"""Generate expected results from the Python engine, for webapp/tests/core.test.js.

Run:  python -P webapp/tests/make_fixtures.py <fixtures.json>

The model is replaced by chosen scores, so every branch of the rule is reached without photos.
"""
from __future__ import annotations

import base64
import io
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11 import weather as wx  # noqa: E402
from stage11.engine import CROP_TF, Engine  # noqa: E402
from stage11.kb import load_kb  # noqa: E402
from stage11.model import EVAL_TF  # noqa: E402


def weather_sets():
    base = {"current_temp_c": 24.0, "current_humidity_pct": 61, "current_precip_mm": 0.0, "week_mean_temp_c": 24.0,
            "week_mean_humidity_pct": 70.5, "week_rain_mm": 3.2, "week_rain_days": 1, "forecast_rain_mm_3d": 0.4,
            "daily": [], "source": "Open-Meteo (open-meteo.com)"}
    return [None, base,
            {**base, "week_mean_temp_c": 31.7, "week_mean_humidity_pct": 88.2, "week_rain_days": 5, "forecast_rain_mm_3d": 9.0, "current_temp_c": 31.2},
            {**base, "week_mean_temp_c": 12.3, "week_mean_humidity_pct": 41.0, "week_rain_days": 0},
            {**base, "week_mean_humidity_pct": None}]


def pixel_fixtures(np_rng) -> list:
    """Smooth and noisy images, resized by the real Pillow and torchvision steps; heat maps by the real OpenCV steps."""
    out = []
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, ::-1]
    for w, h in [(300, 200), (200, 300), (256, 256), (1000, 751), (640, 1280), (257, 999), (225, 224)]:
        noise = np_rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        smooth = np.stack([xx * 255 / w, yy * 255 / h, (xx + yy) * 255 / (w + h)], -1).astype(np.uint8)
        rgb = np.where(np_rng.random((h, w, 1)) < 0.2, noise, smooth)
        img = Image.fromarray(rgb)
        x = EVAL_TF(img).numpy()
        crop = np.array(CROP_TF(img))
        cam = np_rng.normal(0, 1, (7, 7)).astype(np.float32)
        norm = np.maximum(cam, 0)
        norm = norm / (norm.max() + 1e-8)
        heat = cv2.resize(norm, (224, 224), interpolation=cv2.INTER_CUBIC)
        colour = jet[np.uint8(255 * np.clip(heat, 0, 1))]
        overlay = np.uint8(0.55 * crop + 0.45 * colour)
        b64 = lambda a: base64.b64encode(a.tobytes()).decode()
        out.append({"w": w, "h": h, "rgb": b64(rgb), "crop": b64(crop), "input": b64(x.astype(np.float32)),
                    "cam": b64(cam), "overlay": b64(overlay)})
    return out


def main(path: Path) -> None:
    rng = random.Random(11)
    np_rng = np.random.default_rng(11)
    kb = load_kb()
    engine = Engine(device="cpu")
    state = {}
    engine._forward = lambda img: (state["logits"], state["sym"], None)
    engine._gradcam = lambda logits, fmap, cls, img: ""
    wx.fetch = lambda lat, lon: state["weather"]
    wx.geocode = lambda place, st: state["geo"]

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(buf, format="PNG")
    png = buf.getvalue()

    n_cls, n_sym = len(kb.model_classes), len(kb.symptom_ids)
    crops = [d["crop"] for d in kb.model_classes]
    names = []
    for p in kb.plants:
        names += [p["common_name"], p["common_name"] + "s", p["scientific_name"], p["common_name"].upper()]
        names += p.get("aliases", [])[:2]
        if len(p["common_name"]) > 5:
            names.append(p["common_name"][:-1] + "x")  # a typo
    names += ["not a real plant", "zzz", "tomatoe", "potatoe", "appel", "chili"]
    states = [s["name"] for s in kb.states] + ["Jammu", "Hyderabad", "Karnatka", "Nowhere", "Bengal"]
    places = [("", None), ("Mysuru", None), ("Mysuru", {"name": "Mysuru", "district": "Mysore", "state": "Karnataka", "lat": 12.2958, "lon": 76.6394}),
              ("Leh", {"name": "Leh", "district": None, "state": "Ladakh", "lat": 34.1642, "lon": 77.5848})]
    cases = []

    def run(plant, st, place, geo, w, logits, sym):
        state.update(logits=torch.tensor(logits, dtype=torch.float32), sym=np.asarray(sym, dtype=np.float32), weather=w, geo=geo)
        res = engine.diagnose(png, plant, st, place)
        res.pop("heatmap")
        cases.append({"plant": plant, "state": st, "place": place, "geo": geo, "weather": w, "logits": [float(v) for v in state["logits"].numpy()],
                      "sym": [float(v) for v in state["sym"]], "expected": res})

    for i in range(1500):
        plant = rng.choice(names)
        p, _ = kb.resolve_plant(plant)
        lg = np_rng.normal(0, 1.5, n_cls)
        keys = kb.crop_keys(p["model_crop"]) if p else []
        own = [j for j, c in enumerate(crops) if c in keys]
        kind = rng.choice(["own_strong", "own_strong", "own_close", "other_strong", "flat", "mixed"])
        if kind == "own_strong" and own:
            lg[rng.choice(own)] += rng.uniform(4, 12)
        elif kind == "own_close" and own:
            for j in rng.sample(own, min(2, len(own))):
                lg[j] += rng.uniform(3, 5)
        elif kind == "other_strong":
            lg[rng.randrange(n_cls)] += rng.uniform(5, 14)
        elif kind == "mixed" and own:
            lg[rng.choice(own)] += rng.uniform(2, 5)
            lg[rng.randrange(n_cls)] += rng.uniform(2, 5)
        sym = np_rng.beta(0.4, 1.6, n_sym)
        if rng.random() < 0.3:
            target = kb.diseases[rng.randrange(len(kb.diseases))]
            sym = 0.05 + 0.9 * kb.profile(target["key"]) * np_rng.uniform(0.7, 1.0, n_sym)
        place, geo = rng.choice(places)
        run(plant, rng.choice(states), place, geo, rng.choice(weather_sets()), lg, sym)

    # Crops the model may name, with a clear winner, so the named and healthy paths are well covered.
    diagnosable = [p for p in kb.plants if kb.model_coverage(p["model_crop"])["level"] == "diagnosable"]
    for i in range(600):
        p = rng.choice(diagnosable)
        own = [j for j, c in enumerate(crops) if c == p["model_crop"]]
        lg = np_rng.normal(0, 1.0, n_cls)
        lg[rng.choice(own)] += rng.uniform(8, 16)
        place, geo = rng.choice(places)
        run(p["common_name"], rng.choice(states), place, geo, rng.choice(weather_sets()), lg, np_rng.beta(0.4, 1.6, n_sym))

    extra = {
        "resolve": [{"q": q, "name": (kb.resolve_plant(q)[0] or {}).get("common_name"), "match": kb.resolve_plant(q)[1]}
                    for q in names + ["", "   ", "to mato", "Solanum lycopersicum L.", "मिर्च"]],
        "states": [{"q": q, "name": (kb.state(q) or {}).get("name")} for q in states + ["", "West", "Delhi", "Goa"]],
        "coverage": {c: kb.model_coverage(c) for c in sorted({p["model_crop"] for p in kb.plants if p["model_crop"]})},
        "favour": [{"spec": d.get("weather"), "w": w, "out": wx.favourability(d.get("weather"), w)}
                   for d in kb.diseases[:30] for w in weather_sets()],
    }
    path.write_text(json.dumps({"cases": cases, "extra": extra, "pixels": pixel_fixtures(np_rng)}, ensure_ascii=False), encoding="utf-8")
    verdicts = {}
    for c in cases:
        verdicts[c["expected"]["verdict"]] = verdicts.get(c["expected"]["verdict"], 0) + 1
    print(len(cases), "cases", verdicts, "| weather line and reasons covered:",
          sorted({r for c in cases for r in c["expected"]["unknown_reasons"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
