"""Build the static LeafScan site for GitHub Pages into docs/.

Run:  python -P webapp/build_site.py <model.onnx> <folder holding the onnxruntime-web dist files>

The page comes from stage11/static/index.html and the data from stage11/data, so the site follows the app.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.decision import THRESHOLD  # noqa: E402

DOCS = ROOT / "docs"
APP = DOCS / "app"
DATA = ROOT / "stage11" / "data"
VENDOR_FILES = ["ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]
# Text that only makes sense for the local server version of the page.
SITE_TEXT = [
    ("<b>Can't reach the LeafScan server.</b> The state list and plant suggestions come from it.\n"
     "      Start it with <code>python -m stage11.server</code> and open <a href=\"http://localhost:8770\">http://localhost:8770</a>.\n"
     "      Opening <code>index.html</code> directly from disk won't work.",
     "<b>LeafScan's data didn't load.</b> Please check your connection and try again."),
    ("Held-out test sets, each scored once (stage11/results/EVALUATION.md).", "Held-out test sets, each scored once."),
    ("Validation photos only (stage11/results/VAL_THRESHOLD_EVALUATION.md).", "Validation photos only."),
]


def load(name: str):
    return yaml.safe_load((DATA / name).read_text(encoding="utf-8"))


def kb_data() -> dict:
    ckpt = torch.load(ROOT / "models/stage11/best.pt", map_location="cpu", weights_only=False)
    cal = json.loads((ROOT / "models/stage11/calibration.json").read_text())
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, ::-1]
    ev = json.loads((ROOT / "stage11/results/multicrop_eval.json").read_text(encoding="utf-8"))
    return {
        "symptoms": load("symptoms.yaml"), "diseases": load("diseases.yaml"),
        "tests": {t["id"]: t for t in load("diagnostic_tests.yaml")}, "plants": load("plants.yaml"),
        "states": load("india_states.yaml"), "solutions": load("solutions.yaml"), "resources": load("state_resources.yaml"),
        "calibration": {k: cal[k] for k in ("temperature", "beta", "m0")}, "threshold": THRESHOLD,
        "checkpoint_epoch": ckpt["epoch"], "field_eval": ev.get("per_crop", {}), "jet": jet.flatten().tolist(),
    }


def build(model: Path, vendor: Path) -> None:
    (APP / "vendor").mkdir(parents=True, exist_ok=True)
    for name in ("core.js", "pixels.js", "worker.js", "api.js"):
        shutil.copy2(ROOT / "webapp" / name, APP / name)
    for name in VENDOR_FILES:
        shutil.copy2(vendor / "dist" / name, APP / "vendor" / name)
    shutil.copy2(ROOT / "webapp" / "ORT_LICENSE.txt", APP / "vendor" / "ORT_LICENSE.txt")
    if model.resolve() != (APP / "model.onnx").resolve():
        shutil.copy2(model, APP / "model.onnx")
    (APP / "kb.json").write_text(json.dumps(kb_data(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    html = (ROOT / "stage11/static/index.html").read_text(encoding="utf-8")
    assert html.count("<head>") == 1
    html = html.replace("<head>", '<head>\n<script src="app/api.js"></script>')
    for old, new in SITE_TEXT:
        assert html.count(old) == 1, old
        html = html.replace(old, new)
    (DOCS / "index.html").write_text(html, encoding="utf-8", newline="\n")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    for p in sorted(p for p in DOCS.rglob("*") if p.is_file()):
        print(f"{p.stat().st_size / 1e6:8.2f} MB  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
