"""Download only the files LeafScan needs from one pinned commit of the public GitHub repository.

Used while the Hugging Face Space is built:  python fetch_app.py <commit> <output_folder>
Standard library only. The model is checked against its SHA-256, so a changed file stops the build.
"""
from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "Nihar335/open-set-plant-disease"
FILES = {"stage11/__init__.py", "stage11/server.py", "stage11/engine.py", "stage11/kb.py", "stage11/decision.py",
         "stage11/model.py", "stage11/weather.py", "stage11/static/index.html", "stage11/results/multicrop_eval.json",
         "models/stage11/calibration.json"}
FOLDERS = ("stage11/data/",)
MODEL = "models/stage11/best.pt"
MODEL_SHA256 = "a33b763725f6555d3b6dfa5c26828f298baf5df70f18c04a771514200fbe44ba"


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "leafscan-space-build"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch(commit: str, out: Path) -> int:
    count = 0
    archive = download(f"https://codeload.github.com/{REPO}/tar.gz/{commit}")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for m in tar.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if m.isfile() and (rel in FILES or rel.startswith(FOLDERS)):
                target = out / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(m).read())
                count += 1
    missing = [f for f in FILES if not (out / f).is_file()]
    if missing:
        raise SystemExit(f"Missing from the archive: {missing}")

    model = download(f"https://media.githubusercontent.com/media/{REPO}/{commit}/{MODEL}")
    digest = hashlib.sha256(model).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(f"Model hash mismatch: {digest}")
    target = out / MODEL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(model)
    return count + 1


if __name__ == "__main__":
    n = fetch(sys.argv[1], Path(sys.argv[2]))
    print(f"Fetched {n} files, model hash verified.")
