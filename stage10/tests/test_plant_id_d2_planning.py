"""D2/C1 planning tools: metadata-only availability survey and the read-only near-duplicate check (owner approval 2026-10-15)."""
from __future__ import annotations

import inspect
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from stage10.plant_id import neardup as N
from stage10.plant_id.d2 import survey_availability as S

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
CC_BY_NC = "http://creativecommons.org/licenses/by-nc/4.0/"


def _rec(i, who, media_lic=CC_BY, rec_lic=CC_BY + "legalcode", key=111, country="IN"):
    return {"gbifID": i, "speciesKey": key, "countryCode": country, "license": rec_lic, "recordedBy": who,
            "eventDate": "2025-03-01", "references": f"https://www.inaturalist.org/observations/{i}",
            "media": [{"type": "StillImage", "license": media_lic, "creator": who,
                       "identifier": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{i}/original.jpg"}]}


def test_survey_config_taxa_are_disjoint_and_outside_vocabulary():
    cfg = S.config()
    plan = S.taxa_plan(cfg)
    names = [n for n, _, _ in plan]
    assert len(names) == len(set(names)) == 42
    vocab = S.vocabulary_keys()
    unsupported = [n for n, r, _ in plan if r == "unsupported"]
    assert len(unsupported) == 32 and not set(unsupported) & set(vocab)
    assert {n for n, r, _ in plan if r == "supported_control"} <= set(vocab)
    assert set(cfg["c1_species"]) == {"Cocos nucifera", "Areca catechu"}
    m = cfg["minimums"]          # owner 2026-10-15: sample-size requirements are not weakened silently
    assert (m["d2_dev_unsupported_images"], m["d2_dev_unsupported_photographers"], m["d2_test_unsupported_images"]) == (200, 100, 450)
    assert (m["c1_test_per_species"], m["c1_test_photographers_per_species"], m["per_group_min_dev"]) == (60, 30, 30)


def test_survey_requests_only_gbif_metadata_endpoints():
    src = inspect.getsource(S)
    urls = re.findall(r'polite_get\(s, f"\{PS\.GBIF_API\}/([a-z/]+)"', src)
    assert sorted(set(urls)) == ["occurrence/search", "species/match"]
    assert "identifier\"]" not in src.split("def fetch")[1]          # no media URL is ever requested


def test_capped_selection_is_deterministic_and_respects_photographer_cap():
    cands = [{"record_id": str(i), "photographer_key": f"p{i % 3}"} for i in range(20)]
    a = S.capped(cands, 2, "salt")
    assert a == S.capped(list(reversed(cands)), 2, "salt")
    assert len(a) == 6 and max(sum(c["photographer_key"] == p for c in a) for p in ("p0", "p1", "p2")) == 2
    assert len(S.capped(cands, 2, "salt", limit=4)) == 4


def test_analyse_taxon_applies_licence_pilot_and_photographer_rules():
    cfg = S.config()
    recs = [_rec(1, "Alice"), _rec(2, "Alice"), _rec(3, "Alice"),            # 3 by one photographer -> cap 2
            _rec(4, "Bob", media_lic=CC_BY_NC),                               # NC media licence -> excluded
            _rec(5, "Carol", rec_lic=CC_BY_NC + "legalcode"),                 # NC record licence -> excluded
            _rec(6, "Dan"),                                                   # pilot image -> excluded
            _rec(7, "Pilot Person"),                                          # pilot photographer -> development only
            _rec(8, "Eve", key=999),                                          # wrong taxon -> excluded
            _rec(9, "Frank", country="NP")]                                   # not India -> excluded
    t = S.analyse_taxon("X y", "unsupported", 111, recs, 9, cfg, pilot_ids={"6"}, pilot_phot={"pilot person"})
    ex = t["excluded_by_reason"]
    assert ex["record_licence_not_allowed"] == 1 and ex["media_licence_not_allowed"] == 1 and ex["pilot_image"] == 1
    assert ex["taxon_mismatch"] == 1 and ex["not_in_country"] == 1
    assert t["eligible_images_one_per_observation"] == 4 and t["eligible_after_photographer_cap_all_splits"] == 3
    assert t["eligible_from_pilot_photographers_dev_only"] == 1
    p = t["projected_split"]
    assert p["dev"]["available_after_photographer_cap"] + p["test"]["available_after_photographer_cap"] == 3
    assert p["test"]["available_after_photographer_cap"] <= 2           # the pilot photographer never reaches test


def test_pilot_exclusion_sets_cover_all_157_images_and_47_photographers():
    ids, phot = S.pilot_exclusions()
    assert len(ids) == 157 and len(phot) == 47


def test_availability_summary_is_metadata_only_and_complete():
    p = S.OUT / "availability_summary.json"
    if not p.exists():
        pytest.skip("survey not run")
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["nature"].startswith("METADATA-ONLY")
    assert len(s["taxa"]) + len(s["unresolved_names"]) == 42
    assert s["pilot_exclusions"] == {"pilot_images": 157, "pilot_photographers": 47}


# ------------------------------------------------------------------------------------------------ near duplicates
def _img(seed, size=(160, 120)):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return Image.fromarray(a).resize(size, Image.Resampling.BILINEAR)


def test_neardup_detects_transformed_copies_and_separates_distinct_images():
    im = _img(1)
    index_d = np.array([N.dhash(im), N.dhash(_img(2))], dtype=np.uint64)
    index_p = np.array([N.phash(im), N.phash(_img(2))], dtype=np.uint64)
    for name, fn in N.transforms().items():
        if "crop_85" in name:
            continue
        qd, qp = N.query_hashes(fn(im))
        dd, pp = N.min_distances(qd, index_d), N.min_distances(qp, index_p)
        assert min(dd[0], pp[0]) <= 8, name
        assert min(dd[1], pp[1]) > 12, name


def test_neardup_index_and_reports_live_outside_frozen_directories_and_reads_only():
    frozen = {"data", "results", "models", "src", "scripts", "tests", "configs", "paper", "demo"}
    for p in (N.INDEX_DIR, N.REPORTS):
        assert p.relative_to(N.ROOT).parts[0] not in frozen
    src = inspect.getsource(N.open_readonly)
    assert 'open(path, "rb")' in src and "save" not in src


def test_dihedral_variants_make_rotation_distance_zero():
    im = _img(3)
    q, _ = N.query_hashes(im.transpose(Image.Transpose.ROTATE_270))
    assert N.min_distances(q, np.array([N.dhash(im)], dtype=np.uint64))[0] == 0


def test_jpeg_reencode_keeps_phash_close():
    im = _img(4)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=50)
    _, qp = N.query_hashes(Image.open(io.BytesIO(b.getvalue())))
    assert N.min_distances(qp, np.array([N.phash(im)], dtype=np.uint64))[0] <= 6


def test_neardup_reports_record_method_thresholds_and_pairs():
    cal_p, chk_p = N.REPORTS / "calibration.json", N.REPORTS / "pilot_vs_stage1_9.json"
    if not cal_p.exists():
        pytest.skip("near-duplicate check not run")
    cal, chk = json.loads(cal_p.read_text(encoding="utf-8")), json.loads(chk_p.read_text(encoding="utf-8"))
    assert set(cal["thresholds"]) == {"dhash_max_distance", "phash_max_distance"} and cal["seed"] == N.SEED
    assert "centre_crop_85pct" in cal["recall_by_transform_at_thresholds"]       # limitation measured, not hidden
    assert chk["thresholds"] == cal["thresholds"] and chk["index_sha256"] == cal["index_sha256"]
    assert chk["pilot_images_checked"] == 157 and chk["stage1_9_images_indexed"] == 62523
    assert isinstance(chk["flagged_pairs"], list)
