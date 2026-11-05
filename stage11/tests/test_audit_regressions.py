"""Regression tests for the defects confirmed in the 2026-10-23 audit (stage11/audit/AUDIT_REPORT.md).

Run: python -m pytest stage11/tests -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stage11 import weather as wx
from stage11.decision import crop_check, may_name
from stage11.kb import load_kb

ROOT = Path(__file__).resolve().parents[2]
HAS_MODEL = (ROOT / "models/stage11/best.pt").exists()
CASE_PHOTO = ROOT / "stage11_local/case_studies/user_apple_frogeye_1.webp"


# ---------------------------------------------------------------- D1: single-class / related crops were always named
@pytest.mark.parametrize("crop,level", [
    ("apple", "diagnosable"), ("tomato", "diagnosable"), ("maize", "diagnosable"),
    ("soybean", "limited"), ("blueberry", "limited"), ("raspberry", "limited"),  # healthy class only
    ("citrus", "limited"), ("squash", "limited"),                                 # disease class only
    ("chilli", "related"), ("cucurbits", "related"),                              # only a related crop trained
    ("rice", "none"), (None, "none"),
])
def test_model_coverage_levels(crop, level):
    assert load_kb().model_coverage(crop)["level"] == level


def test_coverage_is_derived_from_trained_classes_not_a_list():
    kb = load_kb()
    for crop in {d["crop"] for d in kb.model_classes}:
        cov = kb.model_coverage(crop)
        own = [d for d in kb.model_classes if d["crop"] == crop]
        both = any(d["pathogen_type"] == "none" for d in own) and any(d["pathogen_type"] != "none" for d in own)
        assert (cov["level"] == "diagnosable") == both, crop


@pytest.mark.parametrize("level", ["limited", "related", "none"])
def test_may_name_refuses_partial_coverage_even_at_100_percent(level):
    assert not may_name(1.0, level, conflict=False)


def test_may_name_rules():
    assert may_name(0.60, "diagnosable", False)
    assert not may_name(0.599, "diagnosable", False)
    assert not may_name(0.99, "diagnosable", True)  # crop conflict


# ---------------------------------------------------------------- D2: no explicit crop-conflict check
def test_crop_check_flags_other_crop():
    crops = np.array(["apple", "apple", "potato", "potato"])
    stated = crops == "apple"
    r = crop_check(np.array([0.10, 0.05, 0.80, 0.05]), crops, stated)
    assert r["conflict"] and r["most_likely_crop"] == "potato" and abs(r["stated_crop_mass"] - 0.15) < 1e-9
    assert not crop_check(np.array([0.50, 0.30, 0.15, 0.05]), crops, stated)["conflict"]


def test_crop_check_related_group_is_not_a_conflict():
    crops = np.array(["squash", "tomato"])
    stated = np.isin(crops, load_kb().crop_keys("cucurbits"))
    assert not crop_check(np.array([0.9, 0.1]), crops, stated)["conflict"]


# ---------------------------------------------------------------- server reports coverage honestly
def test_meta_marks_partial_crops_unsupported():
    from stage11.server import app
    meta = app.test_client().get("/api/meta").get_json()
    by = {p["name"]: p for p in meta["plants"]}
    assert by["Tomato"]["supported"] and by["Tomato"]["model_coverage"] == "diagnosable"
    for name in ("Chilli", "Cucumber", "Soybean", "Orange", "Blueberry"):
        assert not by[name]["supported"], name


def test_heic_error_message_names_formats():
    import io
    from stage11.server import app
    r = app.test_client().post("/api/diagnose", data={"plant": "apple", "state": "Goa",
                                                      "image": (io.BytesIO(b"\x00\x00\x00\x18ftypheic"), "x.heic")})
    assert r.status_code == 400 and "JPEG" in r.get_json()["error"]


# ---------------------------------------------------------------- the coverage registry is generated, not edited
def test_coverage_registry_up_to_date():
    from stage11.coverage import OUT_JSON, build
    reg = build()
    assert json.loads(OUT_JSON.read_text(encoding="utf-8")) == json.loads(json.dumps(reg, ensure_ascii=False))
    cats = {p["common_name"]: p["support_category"] for p in reg["plants"]}
    assert cats["Apple"] == "trained" and cats["Soybean"] == "partial" and cats["Rice"] == "symptom_only"


# ---------------------------------------------------------------- engine-level regressions (need the model)
@pytest.fixture(scope="module")
def engine():
    if not HAS_MODEL:
        pytest.skip("model not trained yet")
    from stage11.engine import Engine
    return Engine()


def _val_image(class_key):
    import pandas as pd
    idx = pd.read_csv(ROOT / "stage11/results/image_index.csv")
    return (ROOT / idx[(idx.split == "val") & (idx.class_key == class_key)
                       & (idx.source == "plantvillage")].iloc[0].path).read_bytes()


WET = {"week_mean_temp_c": 22.0, "week_mean_humidity_pct": 95.0, "current_humidity_pct": 95, "current_temp_c": 22,
       "week_rain_days": 6, "forecast_rain_mm_3d": 20.0}
DRY = {"week_mean_temp_c": 38.0, "week_mean_humidity_pct": 15.0, "current_humidity_pct": 15, "current_temp_c": 38,
       "week_rain_days": 0, "forecast_rain_mm_3d": 0.0}


def test_weather_does_not_change_the_score(engine, monkeypatch):
    """D3: the unvalidated weather factor used to multiply the score and could push it across 80%."""
    monkeypatch.setattr(wx, "geocode", lambda place, state: None)
    img = _val_image("potato_late_blight")
    res = []
    for w in (WET, DRY):
        monkeypatch.setattr(wx, "fetch", lambda lat, lon, w=w: w)
        r = engine.diagnose(img, "potato", "Punjab")
        res.append([(c["key"], c["confidence"]) for c in r["candidates"]])
    assert res[0] == res[1]


def test_single_class_crop_never_named(engine, monkeypatch):
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(_val_image("soybean_healthy"), "soybean", "Madhya Pradesh")
    assert r["verdict"] == "unknown" and "model_coverage_limited" in r["unknown_reasons"]
    assert r["plant"]["model_coverage"] == "limited" and not r["plant"]["model_supported"]
    assert r["report"]["fields"]["Predicted disease"].startswith("Unknown disease")  # no candidates to name


def test_related_crop_is_leads_only(engine, monkeypatch):
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(_val_image("pepper_bacterial_spot"), "chilli", "Andhra Pradesh")
    assert r["verdict"] == "unknown" and "model_coverage_related" in r["unknown_reasons"]


def test_unknown_report_states_reasons(engine, monkeypatch):
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(_val_image("soybean_healthy"), "soybean", "Madhya Pradesh")
    text = r["report"]["fields"]["Predicted disease"]
    assert "only part of this crop" in text and "not a diagnosis" in text
    assert r["report"]["fields"]["Test to confirm"]  # never empty, even without candidates


@pytest.mark.skipif(not CASE_PHOTO.exists(), reason="user case photo not present (stage11_local is not shipped)")
def test_user_apple_frogeye_photo_case(engine, monkeypatch):
    """Case study: the user's web photo of apple frog-eye leaf spot (black rot).

    Measured 2026-10-23 with the v2 checkpoint: centre view puts 80% on potato early blight (in-apple mass 15%), so
    the honest result is 'Unknown' with a crop-conflict reason and Black rot among the three leads. This test pins
    that behaviour; it must be revisited (not loosened) if the model is retrained.
    """
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(CASE_PHOTO.read_bytes(), "apple", "Jammu and Kashmir")
    assert r["verdict"] == "unknown"
    assert "crop_conflict" in r["unknown_reasons"]
    assert "apple_black_rot" in [c["key"] for c in r["candidates"]]
    assert all(c["crop"] == "apple" for c in r["candidates"])


# ---------------------------------------------------------------- wrong plant name (user requests 2026-10-23)
DOWNY = ROOT / "stage11_local/case_studies/user_downy_mildew.jpg"


def test_other_crop_is_suggested_not_applied(engine, monkeypatch):
    """A tomato leaf entered as 'squash': the photo's crop is suggested, the name is kept and nothing is named."""
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(_val_image("tomato_early_blight"), "squash", "Karnataka")
    assert r["plant"]["common_name"] == "Squash" and r["verdict"] == "unknown"
    assert r["crop_suggestion"]["suggested"] == "Tomato" and r["crop_suggestion"]["closest_disease"] == "Early blight"
    assert "looks more like Tomato" in r["report"]["fields"]["Plant Name"]
    assert any(n.startswith("Plant check") for n in r["notes"])


def test_correct_plant_name_gets_no_suggestion(engine, monkeypatch):
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(_val_image("tomato_early_blight"), "tomato", "Karnataka")
    assert r["crop_suggestion"] is None and r["verdict"] == "predicted"


def test_untrained_crop_name_gets_suggestion_only(engine, monkeypatch):
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(_val_image("tomato_early_blight"), "brinjal", "Karnataka")
    assert r["plant"]["common_name"] == "Brinjal" and r["verdict"] == "unknown"
    assert r["crop_suggestion"]["suggested"] == "Tomato"


@pytest.mark.skipif(not DOWNY.exists(), reason="user case photo not present (stage11_local is not shipped)")
def test_user_downy_mildew_photo_is_not_switched_to_potato(engine, monkeypatch):
    """Case: the user's downy-mildew web photo entered as tomato. Downy mildew is not a trained class; the model puts
    97% on potato early blight. The automatic crop switch named 'Potato - Early blight 99%'; it must stay unnamed."""
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(DOWNY.read_bytes(), "tomato", "Karnataka")
    assert r["plant"]["common_name"] == "Tomato" and r["verdict"] == "unknown"
    assert r["crop_suggestion"]["suggested"] == "Potato"


@pytest.mark.skipif(not DOWNY.exists(), reason="user case photo not present (stage11_local is not shipped)")
def test_heatmap_uses_models_own_top_class(engine, monkeypatch):
    """The heat map was drawn for the named crop's top class (tomato early blight, ~0% image probability), whose
    only positive evidence was a background corner. It must follow the model's own strongest class."""
    monkeypatch.setattr(wx, "fetch", lambda lat, lon: None)
    r = engine.diagnose(DOWNY.read_bytes(), "tomato", "Karnataka")
    assert r["heatmap_class"] == {"name": "Early blight", "crop": "potato"}
