"""Stage 9 — export the Stage-7 linear head (logistic regression on frozen contrastive features) for inference.

The head was always refitted deterministically from the frozen TRAIN embeddings (scripts/contrastive_osr.py::Fits).
This script performs exactly that fit once and saves coef/intercept to models/contrastive/linear_head.npz so the demo
does not need the embeddings. Verification uses KNOWN VALIDATION only: the recomputed 95% threshold must equal the
frozen Stage-7 threshold. No test data is read. Existing checkpoints are not modified.

  python -P scripts/export_linear_head.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.contrastive_osr import Fits, l2n, load_rep  # noqa: E402
from scripts.osr_msp import sha256  # noqa: E402
from src.osr.scores import energy_score  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def main():
    cfg = load_config()
    f = Fits(cfg, "contrastive", linear_head=True)
    va, Ev = load_rep(cfg, "contrastive", "val")
    k = (va["role"] == "known").to_numpy()
    L = l2n(Ev) @ f.lin.coef_.T + f.lin.intercept_
    t = float(np.quantile(energy_score(L, 1.0)[k], cfg["contrastive"]["target_known_acceptance"], method="higher"))
    saved = json.loads((resolve(cfg["paths"]["metrics"]) / "contrastive_thresholds.json").read_text())["thresholds"]["contrastive::energy_linear_head"]
    if abs(t - saved) > 1e-9:
        raise SystemExit(f"STOP: exported head does not reproduce the Stage-7 threshold ({t} vs {saved})")
    out = resolve("models/contrastive/linear_head.npz")
    np.savez(out, coef=f.lin.coef_, intercept=f.lin.intercept_, classes=np.array(cfg["osr_protocol"]["known_classes"]),
             C=cfg["contrastive"]["linear_head"]["C"])
    print(f"wrote {out.relative_to(ROOT)} sha256={sha256(out)}; reproduces Stage-7 threshold {saved} (abs diff {abs(t - saved):.1e})")


if __name__ == "__main__":
    main()
