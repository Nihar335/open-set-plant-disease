"""Stage 11 tests. Run: python -m pytest stage11/tests -q"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from stage11 import weather as wx
from stage11.kb import load_kb

ROOT = Path(__file__).resolve().parents[2]
HAS_MODEL = (ROOT / "models/stage11/best.pt").exists()


def test_datasets_validate():
    kb = load_kb()
    assert kb.validate() == []
    assert len(kb.model_classes) == 43
    assert len(kb.states) == 36  # 28 states + 8 union territories
    assert len([s for s in kb.states if s["name"] not in {
        "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"}]) == 28


def test_every_disease_has_tests_and_symptoms():
    kb = load_kb()
    for d in kb.diseases:
        assert d["symptoms"], d["key"]
        if d["pathogen_type"] != "none":
            assert d["tests"], d["key"]


@pytest.mark.parametrize("query,expected", [
    ("tomato", "Tomato"), ("Tamatar", "Tomato"), ("aloo", "Potato"), ("corn", "Maize"),
    ("Zea mays", "Maize"), ("bhindi", "Okra"), ("tomatto", "Tomato"), ("paddy", "Rice"),
])
def test_plant_resolution(query, expected):
    p, _ = load_kb().resolve_plant(query)
    assert p and p["common_name"] == expected


def test_unknown_plant():
    assert load_kb().resolve_plant("qwertyuiop") == (None, None)


def test_symptom_targets_shape():
    kb = load_kb()
    t = kb.symptom_targets()
    assert t.shape == (43, len(kb.symptom_ids))
    assert (t.max(1) > 0).all()


WEEK = {"week_mean_temp_c": 18.0, "week_mean_humidity_pct": 92.0, "current_humidity_pct": 90,
        "week_rain_days": 4, "forecast_rain_mm_3d": 5.0}


def test_weather_favourability_late_blight_cool_wet():
    late_blight = load_kb().by_key["potato_late_blight"]["weather"]
    fav = wx.favourability(late_blight, WEEK)
    assert fav["level"] == "high"
    mites = load_kb().by_key["tomato_spider_mites"]["weather"]
    assert wx.favourability(mites, WEEK)["level"] == "low"


def test_weather_factor_bounded():
    for s in (0.0, 0.5, 1.0):
        assert 0.8 - 1e-9 <= wx.factor({"score": s}) <= 1.2 + 1e-9
    assert wx.factor(None) == 1.0


@pytest.fixture(scope="module")
def engine():
    if not HAS_MODEL:
        pytest.skip("model not trained yet")
    from stage11.engine import Engine
    return Engine()


def _val_image(class_key: str) -> bytes:
    import pandas as pd
    idx = pd.read_csv(ROOT / "stage11/results/image_index.csv")
    row = idx[(idx.split == "val") & (idx.class_key == class_key) & (idx.source == "plantvillage")].iloc[0]
    return (ROOT / row.path).read_bytes()


REPORT_KEYS = ["Plant Name", "Biological Name of the plant", "Symptoms found", "Predicted disease",
               "Test to confirm", "Solution for the Predicted disease"]


@pytest.mark.parametrize("name,expected", [("Jammu", "Jammu and Kashmir"), ("tamilnadu", "Tamil Nadu"),
                                           ("Delhi", "Delhi"), ("Pradesh", None)])
def test_state_partial_names(name, expected):
    s = load_kb().state(name)
    assert (s["name"] if s else None) == expected


def test_decision_photo_fit_penalty():
    from stage11.decision import fuse
    kb = load_kb()
    crops = np.array([d["crop"] for d in kb.model_classes])
    mask = crops == "apple"
    logits = np.zeros(len(crops)); logits[[i for i, d in enumerate(kb.model_classes) if d["key"] == "potato_early_blight"][0]] = 5
    logits[1] = 3  # apple_black_rot
    sym = kb.profile("apple_black_rot")
    low = fuse(logits, sym, mask, kb.symptom_targets(), m0=0.5)
    assert low["in_mass"][0] < 0.5 and low["conf"][0].max() < low["fused"][0].max()


def test_engine_supported_crop(engine, monkeypatch):
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)  # offline
    monkeypatch.setattr(wx, "geocode", lambda place, state: None)
    out = engine.diagnose(_val_image("tomato_early_blight"), "tomato", "Karnataka")
    assert out["plant"]["scientific_name"].startswith("Solanum lycopersicum")
    assert out["basis"] == "image"
    assert all(c["crop"] == "tomato" for c in out["candidates"] if c["basis"] != "symptom match only")
    assert out["verdict"] in {"predicted", "unknown", "healthy"}
    assert len(out["candidates"]) == (1 if out["verdict"] != "unknown" else 3)
    assert out["heatmap"].startswith("data:image/png;base64,")
    assert list(out["report"]["fields"]) == REPORT_KEYS
    if out["verdict"] == "predicted":
        c = out["candidates"][0]
        assert c["confidence"] >= 0.60 and c["tests"] and c["solution"]["cultural"]
        assert any("Karnataka" not in x and "University" in x for x in c["where_to_test"])


def test_engine_unsupported_crop_is_unknown(engine, monkeypatch):
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    out = engine.diagnose(_val_image("tomato_early_blight"), "rice", "Punjab")
    assert out["basis"] == "symptom_match"
    assert out["verdict"] == "unknown"
    assert len(out["candidates"]) == 3 and all(c["confidence"] < 0.80 for c in out["candidates"])
    assert out["report"]["fields"]["Predicted disease"].startswith(f"Possibly {out['candidates'][0]['name']}")


def test_server_rejects_missing_fields():
    from stage11.server import app
    c = app.test_client()
    assert c.post("/api/diagnose", data={}).status_code == 400
    r = c.post("/api/diagnose", data={"plant": "tomato", "state": "Goa",
                                      "image": (io.BytesIO(b"not an image"), "x.jpg")})
    assert r.status_code == 400
    meta = c.get("/api/meta").get_json()
    assert "Tamil Nadu" in meta["states"] and any(p["name"] == "Tomato" and p["supported"] for p in meta["plants"])


def test_about_page_numbers_match_current_threshold():
    """The About section quotes stage11/results/val_threshold_eval.json; re-run evaluate_val_threshold.py if the threshold changes."""
    import json
    from stage11.decision import THRESHOLD
    o = json.loads((ROOT / "stage11/results/val_threshold_eval.json").read_text(encoding="utf-8"))
    assert o["threshold"] == THRESHOLD
    assert o["split"].startswith("validation only")
    html = (ROOT / "stage11/static/index.html").read_text(encoding="utf-8")
    f = o["by_source"]["field"]["current"]
    assert f"{100 * f['named']:.1f}%" in html and f"{100 * f['named_acc']:.1f}%" in html


def test_coverage_endpoint():
    from stage11.server import app
    kb = load_kb()
    cov = app.test_client().get("/api/coverage").get_json()
    # every trained class appears exactly once: its disease name, or the crop's healthy flag
    assert sum(len(c["diseases"]) + c["healthy"] for c in cov) == len(kb.model_classes)
    assert len(cov) == len({d["crop"] for d in kb.model_classes})
    tomato = next(c for c in cov if c["crop"] == "tomato")
    assert tomato["level"] == "diagnosable" and tomato["healthy"] and "Septoria leaf spot" in tomato["diseases"]
    squash = next(c for c in cov if c["crop"] == "squash")
    assert squash["level"] == "limited" and not squash["healthy"] and "Pumpkin" in squash["plant_names"]


def test_disease_library_endpoint():
    from stage11.server import app
    kb = load_kb()
    lib = app.test_client().get("/api/diseases").get_json()
    sick = [d for d in kb.diseases if d["pathogen_type"] != "none"]
    assert len(lib) == len(sick)
    assert all(d["pathogen_type"] != "none" for d in lib)
    # photo-trained = disease classes of the image model (43 classes minus the healthy ones)
    assert sum(d["in_model"] for d in lib) == sum(1 for d in kb.model_classes if d["pathogen_type"] != "none")
    scab = next(d for d in lib if d["key"] == "apple_scab")
    assert scab["in_model"] and scab["symptoms"] and scab["tests"] and scab["crop"] == "apple"
