"""Stage-10 must reuse the frozen predictor without changing its behaviour.

Checks (VALIDATION images only; no final-test image is opened):
  * the adapter returns exactly what `OSRPredictor.predict` returns (same object, same numbers);
  * the adapter's numbers equal the existing CLI `scripts/predict.py --json` run in a separate process;
  * the frozen threshold equals the Stage-7 value on disk, and the known/unknown decision follows the frozen rule;
  * adapter sub-objects validate against the Stage-10 report schema;
  * final-test paths are refused by the guard.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys

import pytest

from stage10.guards import ForbiddenImageError, assert_not_final_test
from stage10.osr.osr_adapter import FrozenOSRAdapter, crop_of_class, verify_protected_files
from stage10.schemas.validate import structural_errors
from stage10.tests import fixtures as fx
from src.utils.config import resolve

VAL_IMAGES = [  # both rows are split=val in data/splits/val.csv (asserted below)
    "data/raw/plantvillage/color/Apple___Apple_scab/01f3deaa-6143-4b6c-9c22-620a46d8be04___FREC_Scab 3112.JPG",
    "data/raw/plantvillage/color/Cherry_(including_sour)___healthy/0008f3d3-2f85-4973-be9a-1b520b8b59fc___JR_HL 4092.JPG",
]
pytestmark = pytest.mark.skipif(not all(resolve(p).exists() for p in VAL_IMAGES), reason="PlantVillage not downloaded")


@pytest.fixture(scope="module")
def adapter():
    return FrozenOSRAdapter()


def test_examples_are_validation_images():
    with open(resolve("data/splits/val.csv"), newline="", encoding="utf-8") as f:
        val = {r["image_path"]: r["split"] for r in csv.DictReader(f)}
    assert all(val.get(p) == "val" for p in VAL_IMAGES)
    for p in VAL_IMAGES:
        assert_not_final_test(resolve(p))      # must not raise


def test_protected_stage9_files_unchanged():
    verify_protected_files()


def test_adapter_returns_predictor_output_unchanged(adapter):
    for p in VAL_IMAGES:
        direct = adapter.p.predict(resolve(p))
        wrapped = adapter.assess(resolve(p))["raw"]
        assert wrapped == direct


def test_adapter_matches_existing_cli(adapter):
    out = subprocess.run([sys.executable, "-P", "scripts/predict.py", *VAL_IMAGES, "--json"], cwd=resolve("."),
                         capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    cli = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    assert len(cli) == len(VAL_IMAGES)
    for p, c in zip(VAL_IMAGES, cli):
        a = adapter.assess(resolve(p))["raw"]
        assert a["predicted_known_class"] == c["predicted_known_class"] and a["decision"] == c["decision"]
        for k in ("contrastive_energy", "energy", "msp", "contrastive_energy_threshold", "energy_threshold"):
            assert abs(a[k] - c[k]) < 1e-5, k


def test_frozen_threshold_and_decision_rule(adapter):
    saved = json.loads(resolve("results/metrics/contrastive_thresholds.json").read_text())["thresholds"]["contrastive::energy_linear_head"]
    assert adapter.frozen_threshold == saved
    decisions = []
    for p in VAL_IMAGES:
        o = adapter.assess(resolve(p))["osr_status"]
        assert o["frozen_threshold"] == saved
        assert (o["decision"] == "unknown") == (o["score"] > saved)
        decisions.append(o["decision"])
    assert decisions == ["known", "unknown"]   # recorded Stage-9 behaviour on these two validation images


def test_adapter_objects_fit_the_report_schema(adapter):
    for p, rtype in zip(VAL_IMAGES, ["known_disease", "unknown_investigation"]):
        a = adapter.assess(resolve(p))
        r = fx.known_report() if rtype == "known_disease" else fx.unknown_report(0)
        r["model_prediction"], r["osr_status"] = a["model_prediction"], a["osr_status"]
        assert not structural_errors(r), structural_errors(r)
        assert "not a laboratory confirmation" in r["osr_status"]["interpretation_note"]["en"]
        if rtype == "unknown_investigation":
            assert "Potentially unfamiliar to this model" in r["osr_status"]["message"]["en"]


def test_crop_mapping():
    assert crop_of_class("Corn_(maize)___Common_rust_") == "Corn" and crop_of_class("Pepper,_bell___healthy") == "Pepper bell"
    assert crop_of_class("Cherry_(including_sour)___healthy") is None


def test_guard_refuses_final_test_paths():
    with open(resolve("data/splits/test.csv"), newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))["image_path"]
    with pytest.raises(ForbiddenImageError):
        assert_not_final_test(resolve(first))          # path check only; the image is never opened
    with pytest.raises(ForbiddenImageError):
        FrozenOSRAdapter.assess(object.__new__(FrozenOSRAdapter), resolve(first))
