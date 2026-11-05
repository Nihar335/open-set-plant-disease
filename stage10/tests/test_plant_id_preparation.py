"""Milestone 4 preparation: isolated environment, verified weights, smoke-test record, survey evidence, nothing acquired.

These tests never import open_clip, never load the model and never use the network.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PID = ROOT / "stage10" / "plant_id"
LOCAL = ROOT / "stage10_local"
OFFICIAL_SHA = "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef"
SMOKE = json.loads((PID / "smoke_test_results.json").read_text(encoding="utf-8"))
RECORD = json.loads((PID / "model_record.json").read_text(encoding="utf-8"))


def test_project_environment_has_no_open_clip():
    assert importlib.util.find_spec("open_clip") is None       # model stack lives only in the isolated venv


def test_project_requirements_unchanged_by_plant_id_work():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "open_clip" not in req and "timm" not in req and "huggingface_hub" not in req


def test_pinned_isolated_requirements():
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", (PID / "requirements-plantid-eval.txt").read_text(encoding="utf-8"), re.M))
    assert pins["open_clip_torch"] == "3.3.0" and pins["timm"] == "1.0.30" and pins["huggingface_hub"] == "2.2.0"
    assert len(pins) == 13 and "torch" not in pins           # torch reused from the base interpreter, never reinstalled


def test_weights_record_matches_official_checksum():
    w = RECORD["files"]["open_clip_model.safetensors"]
    assert w["sha256"] == w["official_lfs_sha256"] == OFFICIAL_SHA and w["match"] is True and w["bytes"] == 1710517724
    assert RECORD["revision"] == "2957b322090f9cb17ae72c71981c7218a28d81e0"
    assert "local evaluation only" in RECORD["purpose_approved"]


def test_weights_kept_outside_frozen_and_manifest_scope():
    assert not list((ROOT / "models").rglob("*bioclip*"))
    assert not list((ROOT / "stage10").rglob("*.safetensors"))
    man = json.loads((ROOT / "stage10" / "ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    assert not any(p.startswith("stage10_local") for p in man["stage10_files"])


@pytest.mark.skipif(not (LOCAL / "models" / "bioclip-2" / "open_clip_model.safetensors").exists(), reason="weights not present locally")
def test_local_weights_size_matches_record():
    assert (LOCAL / "models" / "bioclip-2" / "open_clip_model.safetensors").stat().st_size == 1710517724


def test_smoke_test_record():
    assert SMOKE["model"]["sha256_matches_official"] is True and SMOKE["model"]["sha256"] == OFFICIAL_SHA
    assert SMOKE["errors"] == []
    assert "no accuracy" in SMOKE["note"].lower() and "not evidence" in SMOKE["note"].lower()
    flat = json.dumps(SMOKE).lower()
    assert "accuracy\":" not in flat and "top1" not in flat and "top_1" not in flat          # no accuracy was measured
    for k in ("gpu_fp32", "gpu_fp16_autocast", "cpu_fp32"):
        r = SMOKE[k]
        assert r["n_images"] > 0 and r["latency_ms_p95"] >= r["latency_ms_p50"] > 0
    assert SMOKE["environment"]["gpu"] and SMOKE["environment"]["venv"].lower().endswith("agrovaidya-plantid")


def test_pilot_images_on_disk_are_exactly_the_manifested_approved_ones():
    """Acquisition was approved on 2026-10-14 for P1/P2 (P3 blocked): disk content must equal the manifests, nothing more."""
    import csv
    img_dir = LOCAL / "plant_id_pilot" / "images"
    if not img_dir.exists():
        pytest.skip("pilot images not present locally")
    on_disk = {p.relative_to(ROOT).as_posix() for p in img_dir.rglob("*") if p.is_file()}
    listed = set()
    for name in ("p1p2_dev_manifest.csv", "p1p2_sealed_manifest.csv", "p3_sealed_manifest.csv",
                 "p1p2_review_excluded_manifest.csv", "p1p2_quarantined_manifest.csv"):      # review-removed files are preserved
        with open(PID / "pilot" / "manifests" / name, newline="", encoding="utf-8") as f:
            listed |= {r["stored_path"] for r in csv.DictReader(f)}
    assert on_disk == listed
    assert not (img_dir / "P3").exists() or not any((img_dir / "P3").rglob("*.jpg"))     # P3 not acquired (blocked)


@pytest.mark.parametrize("cache", ["results/stage10/plant_id_survey_cache", "results/stage10/fieldcrop_survey_cache"])
def test_survey_evidence_hashes(cache):
    d = ROOT / cache
    lines = [l.split() for l in (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines
    for h, name in lines:
        name = name.lstrip("*")
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == h, name


def test_field_crop_survey_figures_come_from_evidence():
    d = ROOT / "results/stage10/fieldcrop_survey_cache"
    inat = json.loads((d / "survey_inat.json").read_text(encoding="utf-8"))["crops"]
    assert inat["Triticum aestivum"]["2025plus_any_grade"] == 0 and inat["Oryza sativa"]["2025plus_any_grade"] == 5
    summ = {r["id"]: r for r in json.loads((d / "mendeley_summary.json").read_text(encoding="utf-8"))}
    for did in ("8335wpvd7r", "34gb2gr7p2", "bd553pdtny", "zfcv9fmrgv", "nh7zk4hv8z"):   # proposed P3 sources
        assert summ[did]["licence"] == "CC BY 4.0"
    doc = (PID / "FIELD_CROP_SOURCE_SURVEY.md").read_text(encoding="utf-8")
    assert "nothing acquired" in doc and "not a release gate" in doc


def test_docs_do_not_claim_bioclip_works_for_indian_plants():
    for f in PID.glob("*.md"):
        t = f.read_text(encoding="utf-8").lower()
        assert not re.search(r"bioclip[^.]{0,40}(works|is accurate|is validated) for indian", t), f.name
