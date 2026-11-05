"""Negative tests: the validator must FAIL on each injected leakage type.

Run:  python -m pytest tests/test_check_leakage.py -q
Each test copies data/splits to a temp dir, injects one fault, re-locks the copy
(so the specific check, not just the lock, is exercised) and runs the validator.
"""
from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd
import pytest

from src.data import check_leakage
from src.utils.config import PROJECT_ROOT

SPLITS = PROJECT_ROOT / "data" / "splits"
pytestmark = pytest.mark.skipif(not (SPLITS / "manifest_all.csv").exists(), reason="run prepare_dataset.py first")


def _copy(tmp_path):
    d = tmp_path / "splits"
    shutil.copytree(SPLITS, d)
    return d


def _relock(d):
    lock = json.loads((d / "PROTOCOL_LOCK.json").read_text())
    lock["files"] = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in lock["files"]}
    (d / "PROTOCOL_LOCK.json").write_text(json.dumps(lock))


def _rewrite(d, m):
    """Write manifest_all and regenerate the shared split files consistently."""
    m.to_csv(d / "manifest_all.csv", index=False)
    used = m[~m["excluded"].astype(str).str.lower().eq("true")]
    for s in ("train", "val", "test"):
        used[used["split"] == s].to_csv(d / f"{s}.csv", index=False)
        for e in "ABC":
            f = pd.read_csv(d / f"experiment_{e}_{s}.csv", keep_default_na=False)
            keep = f["image_path"].isin(used.loc[used["split"] == s, "image_path"])
            f[keep].to_csv(d / f"experiment_{e}_{s}.csv", index=False)


def _run(d, capsys):
    code = check_leakage.main(["--splits-dir", str(d), "--skip-md5"])
    return code, capsys.readouterr().out


def _manifest(d):
    return pd.read_csv(d / "manifest_all.csv", low_memory=False, keep_default_na=False)


def test_clean_protocol_passes(tmp_path, capsys):
    code, out = _run(_copy(tmp_path), capsys)
    assert code == 0 and out.strip().endswith("PASS")


def test_edit_without_relock_fails(tmp_path, capsys):
    d = _copy(tmp_path)
    m = _manifest(d)
    _rewrite(d, m.iloc[::-1])                       # same content, different bytes
    code, out = _run(d, capsys)
    assert code == 1 and "[FAIL] 1 protocol lock" in out


def test_physical_leaf_leakage_fails(tmp_path, capsys):
    d = _copy(tmp_path)
    m = _manifest(d)
    known = m[(m["role"] == "known") & (m["split"] == "train")]
    g = known.groupby("leaf_group_id").size()
    leaf = g[g > 1].index[0]                        # a leaf with several photos
    row = m.index[m["leaf_group_id"] == leaf][0]
    m.loc[row, "split"] = "test"                    # move ONE photo of that leaf to test
    _rewrite(d, m); _relock(d)
    code, out = _run(d, capsys)
    assert code == 1 and "[FAIL] 5 physical-leaf leakage" in out and "[FAIL] 6 split-group" in out


def test_unknown_class_in_train_fails(tmp_path, capsys):
    d = _copy(tmp_path)
    m = _manifest(d)
    row = m.index[m["role"] == "unknown_near"][0]
    m.loc[row, "split"] = "train"
    _rewrite(d, m); _relock(d)
    code, out = _run(d, capsys)
    assert code == 1 and "[FAIL] 9 unknown-class leakage" in out


def test_unknown_mapped_to_known_id_fails(tmp_path, capsys):
    d = _copy(tmp_path)
    m = _manifest(d)
    m.loc[m.index[m["role"] == "unknown_near"][0], "class_id"] = 3
    _rewrite(d, m); _relock(d)
    code, out = _run(d, capsys)
    assert code == 1 and "[FAIL] 10 label leakage" in out


def test_test_image_duplicated_into_train_fails(tmp_path, capsys):
    d = _copy(tmp_path)
    m = _manifest(d)
    t = m[(m["split"] == "test") & (m["role"] == "known")].iloc[0]
    tr = m.index[(m["split"] == "train") & (m["class_name"] == t["class_name"])][0]
    m.loc[tr, "md5"] = t["md5"]                     # simulate an identical file in train
    _rewrite(d, m); _relock(d)
    code, out = _run(d, capsys)
    assert code == 1 and "[FAIL] 11 test contamination" in out


def test_missing_file_fails(tmp_path, capsys):
    d = _copy(tmp_path)
    m = _manifest(d)
    m.loc[m.index[m["split"] == "train"][0], "image_path"] = "data/raw/does_not_exist.jpg"
    _rewrite(d, m); _relock(d)
    code, out = _run(d, capsys)
    assert code == 1 and "[FAIL] 3 missing files" in out
