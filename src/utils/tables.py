"""CSV + Markdown table writer shared by all scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(to_md(df), encoding="utf-8")
