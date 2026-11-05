"""D2/C1 proposed acquisition manifest (metadata only) and the enforceability of its exclusion and overlap rules.

Owner 2026-10-15: Option B in principle; no acquisition yet. These tests show that the pilot-image and pilot-photographer
exclusions, both caps, the licence allow-list and every overlap check can be enforced, without downloading anything.
"""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest
from PIL import Image

from stage10.plant_id.d2 import plan_manifest as P
from stage10.plant_id.d2 import survey_availability as S
from stage10.plant_id.d2 import validate_plan as V
from stage10.plant_id.pilot import common as PC

HAVE_PLAN = (P.OUT / "d2_proposed_manifest.csv").exists()
HAVE_INDEX = (PC.ROOT / "stage10_local" / "neardup_index" / "stage1_9_hashes.npz").exists()
BY = "http://creativecommons.org/licenses/by/4.0/"
NC = "http://creativecommons.org/licenses/by-nc/4.0/"


def _rec(i, nick, name, media=BY, rec=BY + "legalcode", created="2025-06-01T00:00:00", key=7):
    return {"gbifID": i, "speciesKey": key, "countryCode": "IN", "license": rec, "eventDate": "2025-06-01",
            "http://unknown.org/nick": nick, "recordedBy": name, "stateProvince": "Kerala",
            "references": f"https://www.inaturalist.org/observations/{i}",
            "media": [{"type": "StillImage", "license": media, "creator": name, "created": created,
                       "identifier": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{i}/original.jpg"}]}


def test_frozen_config_restricts_nc_use():
    cfg = P.config()
    assert cfg["status"] == "FROZEN"
    uris = cfg["licences"]["allowed_media_licence_uris"]
    assert NC in uris and len(uris) == 3 and not any(x in u for u in uris for x in ("-sa", "-nd", "3.0", "2.0"))
    text = " ".join(cfg["licences"]["nc_use_restrictions"]).lower()
    for k in ("non-commercial research evaluation", "never redistributed", "train or fine-tune", "deployed or commercial",
              "attribution", "separately", "stop and request owner review"):
        assert k in text, k
    assert cfg["caps"] == {"per_photographer_per_taxon": 2, "per_photographer_per_split": 10}
    assert cfg["split"]["pilot_photographers"] == "development_only" and cfg["c1_status"]["Areca catechu"] == "descriptive_only"


def test_eligibility_rules_licence_versions_identity_and_photo_date():
    cfg = P.config()
    ok, why = P.eligible(_rec(1, "a1", "Asha"), 7, cfg)
    assert why is None and ok["media_licence"] == "CC BY 4.0" and not ok["nc_restricted"]
    nc, _ = P.eligible(_rec(2, "b1", "Bala", media=NC, rec=NC + "legalcode"), 7, cfg)
    assert nc["nc_restricted"] and "CC BY-NC 4.0" in nc["attribution"] and "observations/2" in nc["attribution"]
    assert P.eligible(_rec(3, "c", "C", media="http://creativecommons.org/licenses/by-nc-sa/4.0/"), 7, cfg)[1] == "media_licence_not_allowed"
    assert P.eligible(_rec(4, "d", "D", media="http://creativecommons.org/licenses/by-nc/3.0/"), 7, cfg)[1] == "media_licence_not_allowed"
    assert P.eligible(_rec(5, None, "E"), 7, cfg)[1] == "photographer_unknown"
    assert P.eligible(_rec(6, "f", "F", created="2017-04-23T00:00:00"), 7, cfg)[1] == "photo_created_before_cutoff"
    assert P.eligible(_rec(7, "g", "G", key=8), 7, cfg)[1] == "taxon_mismatch"


def test_identity_clusters_merge_logins_sharing_a_display_name():
    cl = P.identity_clusters([_rec(1, "x1", "Ravi Kumar"), _rec(2, "x2", "ravi  kumar"), _rec(3, "y", "Anu")])
    assert cl["x1"] == cl["x2"] != cl["y"]


def test_selection_respects_both_caps_and_ranks_reserves():
    pools = {"A": [{"gbif_id": str(i), "photographer_key": f"p{i % 2}"} for i in range(30)],
             "B": [{"gbif_id": str(100 + i), "photographer_key": "p0"} for i in range(30)]}
    out = P.select(pools, ["A", "B"], {"A": 3, "B": 3}, 0.5, cap_taxon=2, cap_split=3, salt="s")
    per = {}
    for t, sel in out.items():
        for c, status, rank in sel:
            per[c["photographer_key"]] = per.get(c["photographer_key"], 0) + 1
        assert [r for _, _, r in sel] == list(range(1, len(sel) + 1))
    assert max(per.values()) <= 3 and all(sum(c["photographer_key"] == p for c, _, _ in sel) <= 2 for sel in out.values() for p in per)


@pytest.mark.skipif(not HAVE_PLAN, reason="plan not built")
def test_proposed_manifest_passes_every_rule_and_meets_frozen_minimums_except_areca():
    rs = V.rows()
    assert V.rule_errors(rs) == []
    mins = V.minimums(rs)
    short = {k for k, (_, ok) in mins.items() if not ok}
    assert short == {k for k in mins if k.startswith("C1_Areca")}          # areca: descriptive only, explicitly
    m = S.config()["minimums"]           # frozen minimums unchanged
    assert (m["d2_dev_unsupported_images"], m["d2_test_unsupported_images"], m["c1_test_per_species"]) == (200, 450, 60)


@pytest.mark.skipif(not HAVE_PLAN, reason="plan not built")
def test_proposed_manifest_excludes_pilot_and_keeps_provenance():
    ids, phot = S.pilot_exclusions()
    rs = V.rows()
    assert not {r["gbif_id"] for r in rs} & ids
    assert not {PC.normalise_photographer(r["photographer_display"]) for r in rs if r["split"] == "test"} & phot
    for r in rs:
        assert r["inat_login"] and r["observation_url"] and r["media_licence_uri"] and r["media_created"] >= "2025-01-01"
    s = json.loads((P.OUT / "plan_summary.json").read_text(encoding="utf-8"))
    assert s["nature"].startswith("PROPOSED") and s["held_back_from_test_possible_same_person"] > 0


# --------------------------------------------------------------------------------------------- overlap enforcement
@pytest.mark.skipif(not HAVE_INDEX, reason="near-duplicate index not built")
def test_overlap_gate_rejects_stage1_9_copies_and_pilot_duplicates_and_admits_new_image():
    from stage10.plant_id import neardup as N
    from stage10.plant_id.d2 import overlap_gate as G
    with open(PC.ROOT / PC.config()["prohibited_md5_manifest"], newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    orig = (PC.ROOT / row["image_path"]).read_bytes()                       # read-only
    assert "stage1_9_exact_md5" in G.check(orig, [])[0]
    im = N.open_readonly(PC.ROOT / row["image_path"]).resize((400, 400))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=70)
    assert "stage1_9_near_duplicate" in G.check(b.getvalue(), [])[0]          # resized + re-encoded copy
    with open(PC.HERE / "manifests" / "p1p2_dev_manifest.csv", newline="", encoding="utf-8") as f:
        pr = next(csv.DictReader(f))
    pim = N.open_readonly(PC.ROOT / pr["stored_path"]).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    b = io.BytesIO()
    pim.save(b, format="JPEG", quality=85)
    assert "pilot_near_duplicate" in G.check(b.getvalue(), [])[0]            # flipped re-encoded pilot image
    rng = np.random.default_rng(5)
    new = Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).resize((320, 320), Image.Resampling.BILINEAR)
    b = io.BytesIO()
    new.save(b, format="JPEG", quality=90)
    reasons, ev = G.check(b.getvalue(), [])
    assert reasons == []
    assert "d2_near_duplicate" in G.check(b.getvalue(), [(ev["dhash"], "D2-dev-1")])[0]
    assert "too_small" in G.check(_tiny(), [])[0]


def _tiny():
    b = io.BytesIO()
    Image.new("RGB", (100, 100), (10, 120, 30)).save(b, format="PNG")
    return b.getvalue()
