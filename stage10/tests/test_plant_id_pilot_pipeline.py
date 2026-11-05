"""Pilot acquisition pipeline: offline tests with SYNTHETIC records and images (no network, no project images opened)."""
from __future__ import annotations

import copy
import io
import zipfile

import pytest
from PIL import Image

from stage10.plant_id.pilot import acquire as A
from stage10.plant_id.pilot import common as C
from stage10.plant_id.pilot import sources as S

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"


def jpeg(seed=0, size=(400, 300), flip=False):
    import random
    rnd = random.Random(seed)
    im = Image.new("RGB", size)
    px = im.load()
    for x in range(size[0]):
        for y in range(size[1]):
            px[x, y] = ((x * 7 + seed * 31) % 256, (y * 5 + rnd.randint(0, 3)) % 256, ((x + y) * seed) % 256)
    if flip:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=95)
    return b.getvalue()


def gbif_rec(**over):
    r = {"gbifID": 101, "speciesKey": 3190638, "countryCode": "IN", "eventDate": "2025-03-01", "recordedBy": "Asha K",
         "references": "https://www.inaturalist.org/observations/1", "stateProvince": "Kerala",
         "media": [{"type": "StillImage", "license": CC_BY, "creator": "Asha K", "rightsHolder": "Asha K",
                    "identifier": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/original.jpg"},
                   {"type": "StillImage", "license": CC_BY, "identifier": "https://inaturalist-open-data.s3.amazonaws.com/photos/2/original.jpg"}]}
    r.update(over)
    return r


def cand(rec=None):
    ex = []
    return S.gbif_record_to_candidate("P1", "Mangifera indica", 3190638, rec or gbif_rec(), ex), ex


# ---------------------------------------------------------------- licence / record filters
@pytest.mark.parametrize("uri,ok", [(CC_BY, True), (CC0, True), ("https://creativecommons.org/licenses/by/4.0", True),
                                    ("http://creativecommons.org/licenses/by-nc/4.0/", False),
                                    ("http://creativecommons.org/licenses/by-sa/4.0/", False), (None, False), ("", False)])
def test_licence_whitelist(uri, ok):
    assert C.licence_ok(uri, C.config()["gbif"]["media_licence_uris"]) is ok


def test_good_record_uses_first_photo_medium_variant_and_attribution():
    c, ex = cand()
    assert c and not ex
    assert c.image_url.endswith("/photos/1/medium.jpg") and c.photographer_key == "asha k"
    assert "CC BY 4.0" in c.attribution and "inaturalist.org/observations/1" in c.attribution and "Changes:" in c.attribution


@pytest.mark.parametrize("over,reason", [
    ({"speciesKey": 1, "acceptedTaxonKey": 2, "taxonKey": 3}, "taxon_mismatch"),
    ({"countryCode": "LK"}, "not_in_country"),
    ({"media": []}, "no_still_image"),
    ({"media": [{"type": "StillImage", "license": "http://creativecommons.org/licenses/by-nc/4.0/", "creator": "x",
                 "identifier": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/original.jpg"}]}, "media_licence_not_allowed"),
    ({"media": [{"type": "StillImage", "license": CC_BY, "creator": "x", "identifier": "https://evil.example/a.jpg"}]}, "image_host_not_allowed"),
    ({"recordedBy": None, "media": [{"type": "StillImage", "license": CC_BY,
                                     "identifier": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/original.jpg"}]}, "photographer_unknown"),
])
def test_bad_records_excluded_with_reason(over, reason):
    c, ex = cand(gbif_rec(**over))
    assert c is None and ex and ex[0]["reason"].startswith(reason)


# ---------------------------------------------------------------- split
def test_photographer_split_is_deterministic_and_global():
    s = C.config()["gbif"]["split"]
    keys = [f"user {i}" for i in range(400)]
    a = [C.split_for_photographer(k, s["salt"], s["dev_fraction"]) for k in keys]
    assert a == [C.split_for_photographer(k, s["salt"], s["dev_fraction"]) for k in keys]
    frac = a.count("dev") / len(a)
    assert 0.3 < frac < 0.5


# ---------------------------------------------------------------- images, duplicates, overlap, checksum
@pytest.fixture
def run(tmp_path):
    r = A.Run.__new__(A.Run)
    r.cfg = copy.deepcopy(C.config())
    r.cfg["storage_root"] = str(tmp_path)
    r.bad_md5 = frozenset()
    r.accepted, r.excluded, r.seen = [], [], []
    return r


def mk(record_id="1", expected=None):
    return S.Candidate("P1", "Mangifera indica", "gbif", record_id, "u", "p", "P", "CC BY 4.0", CC_BY, "attr", expected_sha256=expected)


def test_admit_stores_copy_and_full_provenance(run):
    b = jpeg(1)
    assert run.admit(mk(), b, "dev")
    row = run.accepted[0]
    assert row["original_sha256"] == C.sha256(b) and row["original_md5"] == C.md5(b) and row["width"] == 400
    assert C.sha256((C.ROOT / row["stored_path"]).read_bytes()) == row["stored_sha256"]


def test_stage1_9_exact_md5_match_excluded(run):
    b = jpeg(2)
    run.bad_md5 = frozenset({C.md5(b)})
    assert not run.admit(mk(), b, "dev") and run.excluded[0]["reason"] == "exact_md5_match_with_stage1_9_data"


def test_flipped_copy_detected_as_duplicate(run):
    assert run.admit(mk("1"), jpeg(3), "dev")
    assert not run.admit(mk("2"), jpeg(3, flip=True), "dev")
    assert run.excluded[0]["reason"] == "near_duplicate_of:1"
    assert run.admit(mk("3"), jpeg(9), "dev")                     # a different picture is accepted


def test_corrupt_and_small_images_excluded(run):
    assert not run.admit(mk("1"), b"not an image", "dev")
    assert not run.admit(mk("2"), jpeg(4, size=(200, 150)), "dev")
    assert [e["reason"].split(":")[0] for e in run.excluded] == ["corrupt_or_unreadable", "too_small"]


def test_publisher_checksum_mismatch_stops(run):
    with pytest.raises(A.IntegrityStop):
        run.admit(mk(expected="0" * 64), jpeg(5), "sealed")


# ---------------------------------------------------------------- Mendeley
REC = {"name": "Test dataset", "version": 1, "publish_date": "2025-01-01", "doi": {"id": "10.0/x"},
       "contributors": [{"first_name": "A", "last_name": "B"}],
       "data_licence": {"short_name": "CC BY 4.0", "url": "http://creativecommons.org/licenses/by/4.0"}}


def test_mendeley_file_filters_and_stratified_order():
    files = [{"filename": f"{n}.jpg", "id": f"id{n}{fo}", "folder_id": fo, "content_details": {"download_url": "u", "sha256_hash": "h"}}
             for fo in ("f1", "f2", "f3") for n in range(4)]
    files += [{"filename": "x_aug.jpg", "id": "aug", "folder_id": "f1", "content_details": {}},
              {"filename": "notes.json", "id": "js", "folder_id": "f1", "content_details": {}}]
    ex = []
    spec = {"mendeley_id": "abc", "include_folders": ["Keep 1", "Keep 2"]}
    out = S.mendeley_candidates("Zea mays", spec, dict(REC, files=files), {"f1": "Keep 1", "f2": "Keep 2", "f3": "Grain only"}, ex)
    assert {c.folder for c in out} == {"Keep 1", "Keep 2"} and len(out) == 8
    assert [c.folder for c in out[:4]] == ["Keep 1", "Keep 2", "Keep 1", "Keep 2"]          # round-robin across classes
    reasons = {e["reason"].split(":")[0] for e in ex}
    assert reasons == {"folder_not_included", "name_suggests_derived_copy", "not_an_image_file"}
    assert all(c.expected_sha256 == "h" and "Licence CC BY 4.0" in c.attribution and "doi:10.0/x" in c.attribution for c in out)


def test_archive_members_filtered():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("PPLD/Healthy/a.jpg", jpeg(6))
        z.writestr("PPLD/Healthy/a_augmented.jpg", jpeg(6, flip=True))
        z.writestr("PPLD/readme.txt", "x")
    ex = []
    out = S.archive_candidates("Cajanus cajan", {"mendeley_id": "m", "archive": "PPLD.zip"}, REC, buf.getvalue(), ex)
    assert [c.record_id for c, _ in out] == ["PPLD.zip::PPLD/Healthy/a.jpg"]
    assert {e["reason"].split(":")[0] for e in ex} == {"name_suggests_derived_copy", "not_an_image_file"}


def test_non_cc_by_dataset_stops(monkeypatch, tmp_path):
    r = A.Run.__new__(A.Run)

    class Resp:
        status_code = 200
        content = b"{}"

        def json(self):
            return dict(REC, data_licence={"short_name": "CC BY-NC 4.0", "url": "http://creativecommons.org/licenses/by-nc/4.0"})
    monkeypatch.setattr(A, "SOURCE_RECORDS", tmp_path)
    r.get = lambda *a, **k: Resp()
    with pytest.raises(A.IntegrityStop):
        r.mendeley_record("abc")


# ---------------------------------------------------------------- seal
def test_sealed_access_requires_recorded_owner_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "APPROVAL", tmp_path / "SEALED_SCORING_APPROVAL.json")
    with pytest.raises(C.SealedSplitError):
        C.assert_sealed_access_approved()
    (tmp_path / "SEALED_SCORING_APPROVAL.json").write_text('{"approved_by_owner": true}', encoding="utf-8")
    with pytest.raises(C.SealedSplitError):
        C.assert_sealed_access_approved()                           # incomplete record


def test_recorded_approval_is_complete_and_hash_bound():
    """Owner approved exploratory sealed scoring on 2026-10-15; the record must name the exact protocol/amendment/seal hashes."""
    if not C.APPROVAL.exists():
        pytest.skip("no approval recorded")
    rec = C.assert_sealed_access_approved()
    for k in ("protocol_sha256", "amendment_sha256", "vocabulary_sha256", "seal_sha256"):
        assert len(rec[k]) == 64
    assert "EXPLORATORY" in rec["scope"] and rec["date"] == "2026-10-15"


# ---------------------------------------------------------------- retries, rate limits, failures are recorded
class _Resp:
    def __init__(self, code, headers=None):
        self.status_code, self.headers, self.content = code, headers or {}, b""


class _Session:
    def __init__(self, seq):
        self.seq = list(seq)

    def get(self, url, params=None, timeout=None):
        x = self.seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


def test_retries_logged_and_retry_after_honoured(monkeypatch):
    slept = []
    monkeypatch.setattr(S.time, "sleep", lambda s: slept.append(s))
    log = []
    r = S.polite_get(_Session([TimeoutError("t"), _Resp(429, {"Retry-After": "60"}), _Resp(200)]), "u", log=log.append)
    assert r.status_code == 200
    assert [x["outcome"] for x in log] == ["error_TimeoutError", "http_429", "ok_after_retry"]
    assert 60 in slept                                              # server's Retry-After respected, never shortened


def test_not_found_returned_without_retry_and_exhaustion_raises(monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    assert S.polite_get(_Session([_Resp(404)]), "u").status_code == 404
    with pytest.raises(S.DownloadFailed):
        S.polite_get(_Session([_Resp(500)] * 4), "u")


def test_failed_image_download_becomes_recorded_exclusion(run, monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    run.http = _Session([ConnectionError("x")] * 4)
    run.request_log = []
    assert run.fetch_image(mk("9"), sleep=0) is None
    assert run.excluded[-1]["reason"] == "download_failed_after_retries" and len(run.request_log) == 4
