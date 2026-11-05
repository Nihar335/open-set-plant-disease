"""Reproducible dataset acquisition.

Downloads (pinned revisions, see config.yaml):
  * PlantVillage (color)  — maintainer's Hugging Face release, SHA-256 verified,
                            plus the official leaf-grouping map and split lists
  * PlantDoc              — GitHub, pinned commit
  * DTD                   — Oxford VGG (non-plant far-unknown images)

No credentials are needed for the default sources. A Kaggle fallback for
PlantVillage exists (--pv-source kaggle); it reads credentials only from the
KAGGLE_USERNAME / KAGGLE_KEY env vars or ~/.kaggle/kaggle.json.

Usage:
    python scripts/download_dataset.py                       # everything
    python scripts/download_dataset.py --datasets plantdoc   # one dataset
    python scripts/download_dataset.py --keep-archives       # keep zips/tars
    python scripts/download_dataset.py --force               # re-download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config, resolve  # noqa: E402

CHUNK = 1 << 20
DONE_MARKER = ".download_complete.json"


# ----------------------------------------------------------------------------- helpers
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK * 8), b""):
            h.update(block)
    return h.hexdigest()


def http_download(url: str, dest: Path, expected_size: int | None = None,
                  expected_sha256: str | None = None, retries: int = 6) -> Path:
    """Resumable streaming download with optional size/SHA-256 verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_sha256 is None or sha256_of(dest) == expected_sha256:
            print(f"  [cached] {dest.name}")
            return dest
        print(f"  [stale] {dest.name} hash mismatch -> re-downloading")
        dest.unlink()

    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        pos = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if pos and r.status_code != 206:          # server ignored Range
                    pos = 0
                r.raise_for_status()
                total = pos + int(r.headers.get("content-length", 0)) or None
                with open(part, "ab" if pos else "wb") as f, tqdm(
                    total=total, initial=pos, unit="B", unit_scale=True,
                    desc=f"  {dest.name}", leave=True,
                ) as bar:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except (requests.RequestException, OSError) as e:
            print(f"  attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(min(30, 2 ** attempt))

    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{dest.name}: size {size} != expected {expected_size}")
    if expected_sha256 is not None:
        print(f"  verifying SHA-256 of {dest.name} ...")
        got = sha256_of(part)
        if got != expected_sha256:
            part.unlink()
            raise RuntimeError(f"{dest.name}: SHA-256 mismatch ({got})")
    part.replace(dest)
    return dest


def safe_target(base: Path, rel: str) -> Path:
    """Guard against path traversal ("zip slip") in downloaded archives."""
    target = (base / rel).resolve()
    if not target.is_relative_to(base.resolve()):
        raise RuntimeError(f"Unsafe archive member path: {rel}")
    return target


WIN_ILLEGAL = '<>:"|?*'


def portable_name(rel: str) -> str:
    """Replace characters that are illegal in Windows file names (PlantDoc has web
    names like 'IMG_1629.JPG?1507122477.jpg'). Applied on every OS for identical paths."""
    return "/".join("".join("_" if ch in WIN_ILLEGAL or ord(ch) < 32 else ch for ch in part).rstrip(" .")
                    for part in rel.split("/"))


def extract_zip_subtree(zip_path: Path, marker: str, out_dir: Path) -> int:
    """Extract members whose path contains `marker` (e.g. 'raw/color/'),
    re-rooted so that the part *after* the marker lands in out_dir."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if marker in m.filename and not m.is_dir()]
        if not members:
            raise RuntimeError(f"No members containing '{marker}' in {zip_path.name}")
        for m in tqdm(members, desc=f"  extract {marker}", unit="file"):
            rel = m.filename.split(marker, 1)[1]
            target = safe_target(out_dir, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def count_files(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.is_file())


def write_marker(d: Path, info: dict) -> None:
    info["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (d / DONE_MARKER).write_text(json.dumps(info, indent=2))


def is_done(d: Path) -> bool:
    return (d / DONE_MARKER).exists()


# ----------------------------------------------------------------------------- datasets
def download_plantvillage(cfg: dict, paths: dict, args) -> dict:
    c = cfg["datasets"]["plantvillage"]
    out = resolve(c["dir"])
    if is_done(out) and not args.force:
        print("[plantvillage] already present — skipping (use --force to redo)")
        return json.loads((out / DONE_MARKER).read_text())
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{c['repo']}/resolve/{c['revision']}"

    # 1) metadata (leaf grouping + official split lists) — always from HF, small
    meta = out / "meta"
    for f in c["metadata_files"]:
        http_download(f"{base}/{f}", meta / Path(f).name)

    # 2) images
    color_dir = out / c["variant"]
    if args.pv_source == "huggingface":
        zpath = http_download(f"{base}/{c['archive']}", resolve(paths["downloads"]) / "plantvillage_data.zip",
                              expected_size=c["archive_size"], expected_sha256=c["archive_sha256"])
        n = extract_zip_subtree(zpath, f"raw/{c['variant']}/", color_dir)
        archive_info = {"url": f"{base}/{c['archive']}", "sha256": c["archive_sha256"]}
    else:
        zpath, archive_info = kaggle_download(c["kaggle_fallback"], resolve(paths["downloads"]))
        n = extract_zip_subtree(zpath, f"{c['variant']}/", color_dir)
    if not args.keep_archives:
        zpath.unlink()

    info = {"dataset": "plantvillage", "source": args.pv_source, "repo": c["repo"],
            "revision": c["revision"], "archive": archive_info, "variant": c["variant"],
            "n_files_extracted": n, "n_class_dirs": sum(1 for p in color_dir.iterdir() if p.is_dir()),
            "license": c["license"]}
    write_marker(out, info)
    print(f"[plantvillage] {n} files, {info['n_class_dirs']} class folders -> {color_dir}")
    return info


def kaggle_download(slug: str, dl_dir: Path) -> tuple[Path, dict]:
    """Fallback: Kaggle API. Credentials come from env vars or ~/.kaggle/kaggle.json only."""
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env or has_file):
        raise SystemExit("Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or place "
                         "kaggle.json in ~/.kaggle/ (see README). Or use the default HF source.")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("pip install kaggle   (needed only for --pv-source kaggle)")
    api = KaggleApi()
    api.authenticate()
    dl_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(dl_dir), quiet=False, unzip=False)
    zpath = dl_dir / f"{slug.split('/')[-1]}.zip"
    return zpath, {"kaggle": slug, "sha256": sha256_of(zpath)}


def download_plantdoc(cfg: dict, paths: dict, args) -> dict:
    c = cfg["datasets"]["plantdoc"]
    out = resolve(c["dir"])
    if is_done(out) and not args.force:
        print("[plantdoc] already present — skipping")
        return json.loads((out / DONE_MARKER).read_text())
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{c['repo']}/zip/{c['revision']}"
    zpath = http_download(url, resolve(paths["downloads"]) / "plantdoc.zip")
    digest = sha256_of(zpath)
    n, renamed, seen_lower = 0, {}, set()
    with zipfile.ZipFile(zpath) as zf:
        for m in tqdm(zf.infolist(), desc="  extract plantdoc", unit="file"):
            parts = PurePosixPath(m.filename).parts
            if m.is_dir() or len(parts) < 2:
                continue
            rel = "/".join(parts[1:])                      # strip "PlantDoc-Dataset-<sha>/"
            if not (rel.startswith(("train/", "test/")) or rel in ("LICENSE.txt", "README.md")):
                continue
            safe = portable_name(rel)
            # PlantDoc contains distinct images whose names differ only by case
            # (e.g. CAR1.jpg / car1.jpg); on Windows/macOS one would silently overwrite the other.
            k = 1
            while safe.lower() in seen_lower:
                k += 1
                stem, dot, ext = portable_name(rel).rpartition(".")
                safe = f"{stem}__case{k}.{ext}" if dot else f"{ext}__case{k}"
            seen_lower.add(safe.lower())
            if safe != rel:
                renamed[rel] = safe
            target = safe_target(out, safe)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    if not args.keep_archives:
        zpath.unlink()
    info = {"dataset": "plantdoc", "renamed_for_portability": renamed, "repo": c["repo"], "revision": c["revision"], "url": url,
            "archive_sha256": digest, "n_files_extracted": n, "license": c["license"]}
    write_marker(out, info)
    print(f"[plantdoc] {n} files -> {out}")
    return info


def download_dtd(cfg: dict, paths: dict, args) -> dict:
    c = cfg["datasets"]["dtd"]
    out = resolve(c["dir"])
    if is_done(out) and not args.force:
        print("[dtd] already present — skipping")
        return json.loads((out / DONE_MARKER).read_text())
    if args.force and out.exists():
        shutil.rmtree(out)
    tpath = http_download(c["url"], resolve(paths["downloads"]) / "dtd-r1.0.1.tar.gz",
                          expected_size=c["archive_size"])
    digest = sha256_of(tpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tpath, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith("dtd/") and (m.isfile() or m.isdir())]
        tf.extractall(out.parent, members=members, filter="data")   # archive root is dtd/
    if not args.keep_archives:
        tpath.unlink()
    info = {"dataset": "dtd", "url": c["url"], "archive_sha256": digest,
            "n_files_extracted": count_files(out / "images"), "license": c["license"]}
    write_marker(out, info)
    print(f"[dtd] {info['n_files_extracted']} images -> {out / 'images'}")
    return info


DOWNLOADERS = {"plantvillage": download_plantvillage, "plantdoc": download_plantdoc, "dtd": download_dtd}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--datasets", nargs="+", default=list(DOWNLOADERS), choices=list(DOWNLOADERS))
    ap.add_argument("--pv-source", default="huggingface", choices=["huggingface", "kaggle"])
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    free_gb = shutil.disk_usage(resolve(paths["raw"]).anchor).free / 1e9
    print(f"Free disk: {free_gb:.1f} GB (need ~6 GB peak for all datasets)")

    provenance_path = resolve(paths["raw"]) / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    for name in args.datasets:
        print(f"\n=== {name} ===")
        provenance[name] = DOWNLOADERS[name](cfg, paths, args)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2))
    dl = resolve(paths["downloads"])
    if dl.exists() and not any(dl.iterdir()):
        dl.rmdir()
    print(f"\nProvenance written to {provenance_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
