"""Command-line inference with the frozen research prototype (no training, no tuning).

  python -P scripts/predict.py path/to/leaf.jpg [more.jpg ...] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.inference.predictor import OSRPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    p = OSRPredictor()
    for im in a.images:
        r = p.predict(im)
        if a.json:
            print(json.dumps({"image": im, **r}))
            continue
        print(f"\n{im}\n  decision: {r['decision']} — {r['message']}")
        print(f"  closest known disease: {r['predicted_known_class']} (softmax {r['top3'][0][1]:.3f})")
        print(f"  contrastive Energy {r['contrastive_energy']:.3f} (threshold {r['contrastive_energy_threshold']:.3f}; "
              f"{100 * r['contrastive_energy_percentile_known_val']:.1f}th percentile of known validation)")
        print(f"  original Energy {r['energy']:.3f} (threshold {r['energy_threshold']:.3f}) -> {'unknown' if r['energy_says_unknown'] else 'known'}")
    print("\nResearch prototype only — not an agricultural diagnostic system.")


if __name__ == "__main__":
    main()
