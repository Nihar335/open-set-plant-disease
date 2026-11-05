"""Stage-10 integrity: protected Stage-9 files, the Stage-10 manifest, the Stage 1-9 verifier, and the final-test guard."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from stage10.tools.asset_manifest import PROTECTED_STAGE9, check, protected_diffs

ROOT = Path(__file__).resolve().parents[2]


def test_protected_constants_match_baseline_audit():
    audit = (ROOT / "stage10" / "BASELINE_AUDIT.md").read_text(encoding="utf-8")
    for path, h in PROTECTED_STAGE9.items():
        assert re.search(rf"`{re.escape(path)}`\s*\|\s*`{h}`", audit), path


def test_protected_stage9_files_unchanged():
    assert protected_diffs() == []


def test_stage10_manifest_up_to_date():
    assert check() == []


def test_no_stage10_code_references_final_test_manifests():
    allowed = {"stage10/guards.py", "stage10/config.yaml"}
    pat = re.compile(r"test\.csv|experiment_[ABC]_test")
    hits = []
    for p in (ROOT / "stage10").rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and p.suffix in {".py", ".yaml", ".json"} and rel not in allowed and "/tests/" not in f"/{rel}":
            if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
                hits.append(rel)
    assert hits == []


def test_original_frozen_state_verifier_passes():
    out = subprocess.run([sys.executable, "-P", "scripts/verify_final_state.py"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", timeout=900)
    assert out.returncode == 0 and "RESULT: PASS" in out.stdout, out.stdout[-3000:]
