"""Data guard: Stage-10 code must never open a Stage 1-9 final-test image.

The forbidden set is the list of paths in data/splits/test.csv (read as a path list only; no image is opened).
Every Stage-10 entry point that receives a file path calls `assert_not_final_test(path)` first.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ForbiddenImageError(PermissionError):
    pass


def _norm(p) -> str:
    p = Path(p)
    try:
        p = p.resolve().relative_to(ROOT)
    except ValueError:
        pass
    return p.as_posix().lower()


@lru_cache(maxsize=1)
def forbidden_paths() -> frozenset[str]:
    import yaml
    cfg = yaml.safe_load((ROOT / "stage10" / "config.yaml").read_text(encoding="utf-8"))
    man = ROOT / cfg["data_guards"]["forbidden_manifest"]
    with open(man, newline="", encoding="utf-8") as f:
        return frozenset(_norm(ROOT / row["image_path"]) for row in csv.DictReader(f))


def assert_not_final_test(path) -> None:
    if _norm(path) in forbidden_paths():
        raise ForbiddenImageError(f"{path} is a Stage 1-9 final-test image; Stage 10 may not use it")
