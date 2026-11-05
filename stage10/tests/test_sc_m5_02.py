"""SC-M5-02 (owner decision 2026-10-15): photo-first workflow, visible-sign analyser, possible diseases for photos the
frozen model rejects, farmer summary (crop facts, signs, disease/possible diseases, checks, what to do) and the six
interface languages. Validation images only (data/splits/val.csv); no test or held-out image is opened."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image, ImageDraw

from stage10.app import farmer_summary as FS
from stage10.app import i18n_sources as IS
from stage10.app import server as SV
from stage10.knowledge_base.kb import load_kb
from stage10.report_engine.engine import build_report
from stage10.tests import report_fixtures as rf
from stage10.vision import symptoms as SY

ROOT = Path(__file__).resolve().parents[2]
KB = load_kb()
I18N = ROOT / "stage10" / "app" / "i18n"
LANGS = ["en", "hi", "bn", "te", "mr", "ta"]


def leaf_photo(spots=True, seed=0) -> bytes:
    """Drawn test leaf (not a real photo): green ellipse on grey, optionally brown spots with yellow rings."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (640, 480), (125, 125, 125))
    d = ImageDraw.Draw(im)
    d.ellipse((90, 90, 560, 400), fill=(60, 140, 50))
    if spots:
        for _ in range(16):
            x, y = int(rng.integers(170, 470)), int(rng.integers(140, 350))
            d.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(215, 195, 50))
            d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(110, 60, 20))
    arr = np.asarray(im).astype(np.int16) + rng.integers(-18, 19, (480, 640, 1))     # fine texture so the blur gate passes
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=92)
    return b.getvalue()


# ---------------------------------------------------------------------------------------------- visible-sign analyser
def test_calibration_uses_validation_images_only_and_is_recorded():
    cal = SY.calibration()
    assert cal["split_used"].startswith("val") and set(cal["thresholds"]) == set(SY.SIGNS)
    assert "healthy" in cal["rule"] and cal["seen_rate_on_check_half"]


def test_signs_map_only_to_existing_photo_observable_features():
    for sign, feats in SY.SIGN_TO_FEATURES.items():
        assert sign in SY.SIGNS and feats
        for f in feats:
            assert f in KB.features and KB.features[f]["category"] != "environment"


def test_drawn_spotted_leaf_shows_signs_and_clean_leaf_does_not():
    a = SY.analyse(leaf_photo(spots=True))
    assert a.ok and {"brown_spots", "yellow_halo"} <= set(a.seen())
    b = SY.analyse(leaf_photo(spots=False))
    assert b.ok and not {"brown_spots", "dark_spots", "yellow_halo"} & set(b.seen())
    for ob in a.observations():
        assert ob.confidence == "low" and ob.analyser.startswith("colour_shape_v")


def test_no_leaf_is_reported_not_guessed():
    im = Image.new("RGB", (400, 300), (128, 128, 128))
    b = io.BytesIO()
    im.save(b, "PNG")
    a = SY.analyse(b.getvalue())
    assert not a.ok and "no leaf" in a.reason and a.observations() == ()


def test_healthy_validation_leaves_rarely_show_disease_signs():
    v = pd.read_csv(ROOT / "data/splits/val.csv", low_memory=False)
    assert set(v.split) == {"val"}
    healthy = v[v.class_name.str.endswith("healthy")].groupby("class_name").head(3)
    sick = v[v.class_name.isin(["Tomato___Early_blight", "Potato___Early_blight", "Corn_(maize)___Common_rust_"])].groupby("class_name").head(3)
    hs = [len(set(SY.analyse(ROOT / p).seen()) & {"brown_spots", "yellow_halo", "yellowing"}) for p in healthy.image_path]
    ss = [len(set(SY.analyse(ROOT / p).seen()) & {"brown_spots", "yellow_halo", "yellowing"}) for p in sick.image_path]
    assert np.mean(ss) > np.mean(hs)


# ---------------------------------------------------------------------------------------------- rejected photo -> possible diseases
def test_rejected_photo_gives_ranked_possible_diseases_not_insufficient_evidence():
    a = rf.synthetic_assessment("Tomato___Early_blight", False,
                                top3=[("Tomato___Early_blight", 0.55), ("Tomato___Late_blight", 0.25), ("Tomato___Bacterial_spot", 0.1)])
    req = rf.request(plant_name="Tomato", plant_confirmation="confirmed", model_assessment=a)
    res = build_report(req, KB)
    r = res.report
    assert r["report_type"] == "unknown_investigation"            # build_report already ran the engine's own checks
    ids = [c["kb_disease_id"] for c in r["candidate_diagnoses"]]
    assert ids[0] == "COND-TOMATO-EARLY-BLIGHT" and "COND-TOMATO-LATE-BLIGHT" in ids and len(ids) <= 3
    assert all(c["ranking_method"] in ("hybrid", "kb_symptom_match") and c["ranking_score_is_probability"] is False for c in r["candidate_diagnoses"])
    assert r.get("insufficient_evidence_statement") in (None, {}) or not r["insufficient_evidence_statement"]


def test_other_crops_and_healthy_classes_never_become_possible_diseases():
    a = rf.synthetic_assessment("Potato___Late_blight", False, top3=[("Potato___Late_blight", 0.6), ("Tomato___healthy", 0.3)])
    req = rf.request(plant_name="Tomato", plant_confirmation="confirmed", model_assessment=a)
    ids = [c["kb_disease_id"] for c in build_report(req, KB).report["candidate_diagnoses"]]
    assert "COND-POTATO-LATE-BLIGHT" not in ids


def test_every_frozen_model_disease_class_now_maps_to_a_sourced_record():
    for m in KB.raw["model_classes"]:
        if m["research_role"] == "frozen_model_known_class" and m["condition_id"]:
            assert KB.conditions[m["condition_id"]]["curation_status"] == "sourced", m["class_name"]


# ---------------------------------------------------------------------------------------------- farmer summary
@pytest.mark.parametrize("pid", sorted(KB.plants))
def test_crop_facts_are_sourced_or_explicitly_missing(pid):
    c = FS.crop_facts(KB, pid, None)
    assert c["scientific_name"] and c["family"]
    for k in ("temperature", "soil", "nutrients"):
        assert c[k] is None or (c[k]["text"] and c[k]["sources"])
    assert c["temperature"] is not None                      # all 8 crops have a sourced temperature statement


class FakeOSR:
    def __init__(self, cls="Tomato___Early_blight", known=False, top3=None):
        self.cls, self.known, self.top3 = cls, known, top3

    def assess(self, image):
        return rf.synthetic_assessment(self.cls, self.known, self.top3 or [(self.cls, 0.6), ("Tomato___Late_blight", 0.3)])


def run(factory, plant="Tomato", img=None):
    c = SV.create_app(osr_factory=factory, kb=KB).test_client()
    data = {"plant": json.dumps({"action": "typed", "text": plant, "confirmed": True}), "language": "en",
            "image": (io.BytesIO(img or leaf_photo()), "leaf.jpg")}
    j = c.post("/api/v1/assess", data=data, content_type="multipart/form-data").get_json()
    assert j["ok"], j
    return j["data"]


def test_api_rejected_photo_summary_has_every_owner_field():
    d = run(FakeOSR)
    s = d["summary"]
    assert s["kind"] == "possible" and 1 <= len(s["diseases"]) <= 3 and s["model"]["accepted"] is False
    assert s["crop"]["name"] == "Tomato" and s["crop"]["scientific_name"].startswith("Solanum lycopersicum")
    assert s["crop"]["temperature"] and s["crop"]["soil"] and s["crop"]["nutrients"]["needed"]
    assert {"brown_spots", "yellow_halo"} <= set(s["signs"]["seen"])
    for dz in s["diseases"]:
        assert dz["checks"] and (dz["what_to_do"] or dz["sprays_named_by_source"]) and dz["typical_signs"]
    # image-found signs are marked inferred in the full report, never as user observations
    img = [v for v in d["report"]["visible_symptoms"] if v["origin"] == "image_analysis"]
    assert img and all(v["evidence_status"] == "inferred" for v in img)


def test_api_accepted_photo_summary_names_the_disease():
    s = run(lambda: FakeOSR(known=True, top3=[("Tomato___Early_blight", 0.95)]))["summary"]
    assert s["kind"] == "known" and s["diseases"][0]["condition_id"] == "COND-TOMATO-EARLY-BLIGHT"
    assert any("rotation" in x["text"].lower() or "debris" in x["text"].lower() for x in s["diseases"][0]["what_to_do"])


def test_api_healthy_class_is_healthy_looking_not_a_disease():
    s = run(lambda: FakeOSR(cls="Tomato___healthy", known=True, top3=[("Tomato___healthy", 0.97)]), img=leaf_photo(spots=False))["summary"]
    assert s["kind"] == "healthy_looking" and not s["diseases"] and "does not prove" in s["notes"][0]


def test_summary_never_states_a_dose():
    s = run(FakeOSR)["summary"]
    dose = re.compile(r"\d+(\.\d+)?\s*(%|kg/ha|g/l|ml/l|ppm)", re.I)
    assert not dose.search(json.dumps(s))


# ---------------------------------------------------------------------------------------------- languages
def test_exactly_six_interface_languages_are_offered():
    cfg = yaml.safe_load((ROOT / "stage10/config.yaml").read_text(encoding="utf-8"))
    assert [x["code"] for x in cfg["language"]["ui_languages"]] == LANGS
    c = SV.create_app(osr_factory=FakeOSR, kb=KB).test_client()
    assert [x["code"] for x in c.get("/api/v1/health").get_json()["data"]["ui_languages"]] == LANGS
    assert c.get("/api/v1/i18n/gu").status_code == 404


@pytest.mark.parametrize("lang", LANGS[1:])
def test_language_catalogue_is_complete_and_labelled_as_unreviewed_draft(lang):
    en = json.loads((I18N / "en.json").read_text(encoding="utf-8"))["strings"]
    cat = json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))
    assert cat["meta"]["review_status"] == "machine_draft_unreviewed" and "NOT checked" in cat["meta"]["note"]
    assert set(cat["strings"]) == set(en)
    for k, v in cat["strings"].items():
        assert set(re.findall(r"\{(\w+)\}", v)) == set(re.findall(r"\{(\w+)\}", en[k])), k
    assert set(IS.server_strings()) <= set(cat["kb"])
    built, err = IS.build_one(lang, {})
    assert not err and built["strings"] == cat["strings"]       # catalogue is up to date with its draft


def test_server_text_list_is_current():
    assert json.loads(IS.OUT.read_text(encoding="utf-8"))["strings"] == IS.server_strings()
