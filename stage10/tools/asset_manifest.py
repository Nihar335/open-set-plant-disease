"""Stage-10 asset manifest (separate from the authoritative Stage 1-9 inventory, which this tool never touches).

  python -m stage10.tools.asset_manifest --check   # verify protected Stage-9 files + recorded Stage-10 files
  python -m stage10.tools.asset_manifest --write   # (re)record Stage-10 file hashes + dependency versions

Sections of stage10/ASSET_MANIFEST.json:
  protected_stage9_files  baseline hashes recorded in stage10/BASELINE_AUDIT.md (2026-10-14). They are constants in this
                          file; --write refuses to run if any of them changed, so a change can never be "absorbed".
  stage10_files           every file under stage10/ (+ Stage-10 scripts/apps when they exist), excluding caches/outputs.
  dependencies            installed versions of packages Stage 10 relies on, plus approved-but-not-yet-installed ones.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "stage10" / "ASSET_MANIFEST.json"

PROTECTED_STAGE9 = {  # from stage10/BASELINE_AUDIT.md §1 — never edit
    "models/contrastive/linear_head.npz": "7a9fbe86958114c6bfff0eef439b688fd68e4d24a3846d946e61238509100a4b",
    "src/inference/predictor.py": "bb246f80d5b00821079b4b4ef3ca5d4a87c50997f524513f30765d113330afcb",
    "scripts/predict.py": "adb8f17927d03e51674cfa3ca2613c091f79e33dbb61d37988866e76037b28bf",
    "demo/app.py": "e37364e1ac108955e87e442d6753912c37f9eb1ce543b9107a94a5af61a3b15b",
    "scripts/verify_final_state.py": "b1fc44fb70150e48ab3dacd8ad4fe0d9ef32f4784dcc7d39c24a0eee37afcd38",
    "results/final/final_state_inventory.json": "f0ca9dd48ad36b7cdcbae272f09b26b0bba6510dfe6fdeb9b1edf40b027aa6ff",
}
STAGE10_GLOBS = ["stage10/**/*", "demo/agrovaidya_app.py", "scripts/*stage10*.py"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache"}
PACKAGES = ["torch", "torchvision", "numpy", "Pillow", "opencv-python", "PyYAML", "jsonschema", "referencing", "pydantic",
            "streamlit", "pytest", "scikit-learn", "pandas", "Flask", "Werkzeug"]   # Flask: M5 app (already installed)
APPROVED_NOT_YET_INSTALLED = {"fpdf2": "approved 2026-10-14 for the PDF milestone (pin on install)",
                              "uharfbuzz": "approved 2026-10-14 for Devanagari text shaping (pin on install)"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage10_files() -> dict:
    out = {}
    for g in STAGE10_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p.is_file() and p != MANIFEST and not (set(p.relative_to(ROOT).parts) & EXCLUDE_PARTS):
                out[p.relative_to(ROOT).as_posix()] = sha256(p)
    return out


def dependencies() -> dict:
    inst = {}
    for n in PACKAGES:
        try:
            inst[n] = md.version(n)
        except md.PackageNotFoundError:
            inst[n] = None
    planned = {}
    for n, note in APPROVED_NOT_YET_INSTALLED.items():
        try:
            planned[n] = {"installed": md.version(n), "note": note}
        except md.PackageNotFoundError:
            planned[n] = {"installed": None, "note": note}
    return {"python": platform.python_version(), "installed": inst, "approved_additions": planned}


def protected_diffs() -> list[str]:
    return [p for p, h in PROTECTED_STAGE9.items() if not (ROOT / p).exists() or sha256(ROOT / p) != h]


def check() -> list[str]:
    problems = [f"PROTECTED CHANGED: {p}" for p in protected_diffs()]
    if not MANIFEST.exists():
        return problems + ["manifest missing (run --write)"]
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if man["protected_stage9_files"] != PROTECTED_STAGE9:
        problems.append("manifest protected section differs from the audited constants")
    cur = stage10_files()
    for p, h in man["stage10_files"].items():
        if p not in cur:
            problems.append(f"stage10 file missing: {p}")
        elif cur[p] != h:
            problems.append(f"stage10 file changed since last --write: {p}")
    problems += [f"stage10 file not yet recorded: {p}" for p in cur if p not in man["stage10_files"]]
    return problems


def write() -> None:
    bad = protected_diffs()
    if bad:
        raise SystemExit(f"REFUSING to write: protected Stage-9 files changed {bad}. Investigate; do not absorb the change.")
    man = {"description": "Stage-10 (AgroVaidya) asset manifest. The Stage 1-9 inventory results/final/final_state_inventory.json "
                          "remains authoritative for the frozen research artifacts.",
           "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protected_stage9_files": PROTECTED_STAGE9, "stage10_files": stage10_files(), "dependencies": dependencies()}
    MANIFEST.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(man['stage10_files'])} Stage-10 files, {len(PROTECTED_STAGE9)} protected files OK")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        write()
        return
    probs = check()
    for p in probs:
        print(" ", p)
    print("STAGE10 MANIFEST:", "PASS" if not probs else "FAIL")
    sys.exit(0 if not probs else 1)


if __name__ == "__main__":
    main()
