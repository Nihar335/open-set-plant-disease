"""Access to the frozen Stage-2 protocol: lock verification and filtered manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from src.utils.config import resolve


class ProtocolError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_lock(cfg: dict) -> dict:
    """Raise ProtocolError unless every locked manifest and the protocol config are unchanged."""
    sd = resolve(cfg["paths"]["splits"])
    lock = json.loads((sd / "PROTOCOL_LOCK.json").read_text())
    changed = [f for f, h in lock["files"].items() if not (sd / f).exists() or _sha256(sd / f) != h]
    op_hash = hashlib.sha256(json.dumps(cfg["osr_protocol"], sort_keys=True).encode()).hexdigest()
    if changed or op_hash != lock["protocol_config_sha256"]:
        raise ProtocolError(f"Locked protocol changed: files={changed}, "
                            f"protocol_config_changed={op_hash != lock['protocol_config_sha256']}")
    return lock


def known_classes(cfg: dict) -> list[str]:
    """Class names in class-id order, cross-checked against class_mapping.yaml."""
    names = list(cfg["osr_protocol"]["known_classes"])
    cm = yaml.safe_load(resolve(cfg["baseline"]["dataset"]["class_mapping"]).read_text(encoding="utf-8"))
    mapped = [c["class_name"] for c in sorted(cm["known_classes"], key=lambda c: c["class_id"])]
    if mapped != names:
        raise ProtocolError("class_mapping.yaml order differs from config osr_protocol.known_classes")
    return names


def load_manifest(path: str | Path, row_filter: dict, split: str, classes: list[str]) -> pd.DataFrame:
    """Read a locked manifest and keep only rows matching `row_filter` (closed-set: known classes).

    Asserts the closed-set invariants so an unknown image can never slip in silently."""
    df = pd.read_csv(resolve(path), low_memory=False, keep_default_na=False)
    for col, allowed in row_filter.items():
        df = df[df[col].isin(allowed)]
    df = df[df["split"] == split].reset_index(drop=True)
    df["class_id"] = df["class_id"].astype(int)
    df["strict_test"] = df["strict_test"].astype(str).str.lower().eq("true")
    if (df["known_unknown"] != "known").any():
        raise ProtocolError(f"{split}: non-known rows after filtering")
    if not df["class_id"].between(0, len(classes) - 1).all():
        raise ProtocolError(f"{split}: class_id out of range")
    if (df["class_name"].map({c: i for i, c in enumerate(classes)}) != df["class_id"]).any():
        raise ProtocolError(f"{split}: class_id does not match class_name")
    return df
