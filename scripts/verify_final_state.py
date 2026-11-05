"""Stage 9 — verify the frozen state of the project (read-only; never scores the test set).

  python -P scripts/verify_final_state.py            # verify against results/final/final_state_inventory.json
  python -P scripts/verify_final_state.py --write    # (once) create the inventory with sha256 of frozen artifacts

Checks: protocol lock + split hashes, leakage validator, checkpoint hashes, metric-file hashes (vs the
hashes Stage 8 recorded and vs this inventory), cross-stage metric consistency, validation-only thresholds
(created before the first test-log entry of their stage), zero unknown images in training, and that the
required caveat labels are still present in the reports.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.osr_msp import sha256  # noqa: E402
from src.data.protocol import verify_lock  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

INV = "results/final/final_state_inventory.json"
FROZEN_GLOBS = ["data/splits/*.csv", "data/splits/*.json", "data/splits/*.yaml", "models/*/*.pt", "models/*/*.yaml",
                "models/*/*.json", "results/metrics/*", "results/predictions/*.csv", "results/embeddings/*",
                "results/embeddings_contrastive/*", "results/predictions/cache/*", "results/test_runs.log", "config.yaml"]
STAGE_THRESHOLDS = [("stage4", "osr_msp_threshold.json"), ("stage5", "osr_energy_threshold.json"),
                    ("stage6", "mahalanobis_threshold.json"), ("stage7", "contrastive_thresholds.json"),
                    ("stage7 POST-HOC", "contrastive_head_control_thresholds.json"), ("stage8", "fusion_threshold.json")]


def frozen_hashes():
    out = {}
    for g in FROZEN_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p.is_file():
                out[p.relative_to(ROOT).as_posix()] = sha256(p)
    return out


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    M = resolve(cfg["paths"]["metrics"])
    checks, notes = {}, []

    # 1. protocol lock and leakage validator
    lock = verify_lock(cfg)
    checks["protocol_lock_valid_and_splits_unchanged"] = True
    r = subprocess.run([sys.executable, "-m", "src.data.check_leakage"], cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    checks["leakage_validator_PASS"] = r.returncode == 0 and r.stdout.strip().splitlines()[-1].strip() == "PASS"

    # 2. checkpoints
    env7 = json.loads((M / "contrastive_environment.json").read_text())
    checks["stage3_checkpoint_matches_config_sha"] = sha256(resolve(cfg["osr_msp"]["checkpoint"])) == cfg["osr_msp"]["checkpoint_sha256"]
    checks["contrastive_checkpoint_matches_recorded_sha"] = sha256(resolve("models/contrastive/best_model.pt")) == env7["checkpoint_sha256"]

    # 3. metric files unchanged since Stage 8 recorded them
    f8 = json.loads((M / "fusion_threshold.json").read_text())
    changed = [k for k, v in f8["prior_metric_files_sha256"].items() if not (M / k).exists() or sha256(M / k) != v]
    checks["pre_stage8_metric_files_unchanged"] = not changed
    if changed:
        notes.append(f"changed metric files: {changed}")

    # 4. no unknown images in training
    tr = pd.read_csv(resolve("data/splits/train.csv"), low_memory=False, keep_default_na=False)
    rf = cfg["baseline"]["dataset"]["row_filter"]
    sel = tr
    for k, v in rf.items():
        sel = sel[sel[k].isin(v if isinstance(v, list) else [v])]
    forbidden = set(cfg["osr_protocol"]["unknown_near"]) | set(cfg["osr_protocol"]["unknown_val"])
    m7 = pd.read_csv(resolve("results/embeddings_contrastive/train_meta.csv"), low_memory=False)
    m3 = pd.read_csv(resolve("results/embeddings/train_meta.csv"), low_memory=False)
    checks["no_unknown_images_in_training"] = (set(sel["role"]) == {"known"} and not set(sel["class_name"]) & forbidden
                                               and set(m7["role"]) == {"known"} and set(m3["role"]) == {"known"})

    # 5. validation-only thresholds: created before the first test-log entry of their stage
    log = resolve("results/test_runs.log").read_text(encoding="utf-8").splitlines()
    first = {}
    for line in log:
        ts, rest = line.split(" ", 1)
        for tag, _ in STAGE_THRESHOLDS:
            key = tag.split()[0]
            if rest.startswith(tag + " ") and "NOTE" not in rest.split()[1] and tag not in first:
                if tag == "stage7" and "POST-HOC" in rest:
                    continue
                first[tag] = iso(ts)
    thr = {}
    for tag, f in STAGE_THRESHOLDS:
        d = json.loads((M / f).read_text())
        ok = d.get("test_used") is False and iso(d["created_utc"]) <= first[tag]
        thr[tag] = {"file": f, "created_utc": d["created_utc"], "first_test_log": first[tag].isoformat(), "test_used": d.get("test_used"), "ok": ok}
    checks["thresholds_frozen_before_test_scoring"] = all(v["ok"] for v in thr.values())
    checks["test_runs_log_entries"] = len(log)
    crashed = [l[:120] for l in log if "NOTE" in l]

    # 6. cross-stage consistency of the primary numbers
    s4 = json.loads((M / "osr_msp_results.json").read_text())["results"]["B_hard_same_crop"]["auroc"]
    s5 = json.loads((M / "osr_energy_results.json").read_text())["energy_results"]["primary"]["B_hard_same_crop"]["auroc"]
    s6 = json.loads((M / "mahalanobis_results.json").read_text())
    s7 = json.loads((M / "contrastive_results.json").read_text())
    s8 = json.loads((M / "fusion_results.json").read_text())["results"]
    pairs = {"MSP s4=s6=s8": [s4, s6["msp_results"]["B_hard_same_crop"]["auroc"], s8["MSP"]["B_hard_same_crop"]["auroc"]],
             "Energy s5=s6=s7=s8": [s5, s6["energy_results"]["B_hard_same_crop"]["auroc"], s7["prior_original_results"]["original::energy"]["B_hard_same_crop"]["auroc"],
                                    s8["Energy"]["B_hard_same_crop"]["auroc"]],
             "Mahalanobis s6=s8": [s6["mahalanobis_results"]["B_hard_same_crop"]["auroc"], s8["Mahalanobis"]["B_hard_same_crop"]["auroc"]],
             "Contrastive Energy s7=s8": [s7["results"]["contrastive::energy_linear_head"]["B_hard_same_crop"]["auroc"], s8["Contrastive Energy*"]["B_hard_same_crop"]["auroc"]]}
    cons = {k: {"values": v, "consistent": max(v) - min(v) < 1e-12} for k, v in pairs.items()}
    checks["cross_stage_metrics_consistent"] = all(x["consistent"] for x in cons.values())

    # 7. caveat labels still present in reports
    rep = lambda n: (ROOT / "results/reports" / n).read_text(encoding="utf-8")  # noqa: E731
    labels = {
        "stage6_probe_labelled_diagnostic": bool(re.search(r"not an OSR method|diagnostic", rep("STAGE_6_FEATURE_OSR.md"), re.I)),
        "stage7_probe_labelled_not_OSR": "not an OSR method" in rep("STAGE_7_REPRESENTATION_LEARNING.md"),
        "stage7_head_control_labelled_post_hoc": "Head-matched control (post-hoc)" in rep("STAGE_7_REPRESENTATION_LEARNING.md")
                                                 and "POST-HOC" in json.loads((M / "contrastive_head_control.json").read_text())["note"],
        "stage8_test_selection_caveat_explicit": "**post hoc**" in rep("STAGE_8_FINAL_OSR_BENCHMARK.md"),
        "stage8_oracle_labelled": "ORACLE ANALYSIS — NOT PART OF THE DEPLOYABLE OSR METHOD" in rep("STAGE_8_FINAL_OSR_BENCHMARK.md"),
    }
    checks.update(labels)

    # 8. hashes vs inventory
    H = frozen_hashes()
    inv_path = ROOT / INV
    if a.write or not inv_path.exists():
        inv = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "purpose": "Frozen state of the project at the end of Stage 9 (Stages 1-8 complete; no further experiments)",
               "protocol_lock": {"created_utc": lock["created_utc"], "protocol_config_sha256": lock["protocol_config_sha256"],
                                 "files": lock["files"]},
               "checkpoints": {"stage3_baseline": {"path": cfg["osr_msp"]["checkpoint"], "sha256": cfg["osr_msp"]["checkpoint_sha256"]},
                               "stage7_contrastive": {"path": "models/contrastive/best_model.pt", "sha256": env7["checkpoint_sha256"],
                                                      "best_epoch": env7.get("best_epoch")}},
               "thresholds": thr, "test_runs_log": log, "crash_notes_in_test_log": crashed,
               "cross_stage_consistency": cons, "reports": sorted(p.name for p in (ROOT / "results/reports").glob("*.md")),
               "stage_1_2_documentation": ["DATASET_CARD.md", "README.md (Stages 1-2 sections)", "results/metrics/dataset_inventory.json",
                                           "results/metrics/leakage_report.json", "results/check_leakage_final.log"],
               "frozen_artifact_sha256": H}
        inv_path.parent.mkdir(parents=True, exist_ok=True)
        inv_path.write_text(json.dumps(inv, indent=2, default=str))
        print(f"wrote {INV} ({len(H)} hashed artifacts)")
    inv = json.loads(inv_path.read_text())
    drift = sorted(k for k, v in inv["frozen_artifact_sha256"].items() if H.get(k) != v)
    new = sorted(k for k in H if k not in inv["frozen_artifact_sha256"])
    checks["frozen_artifacts_match_inventory"] = not drift
    if drift:
        notes.append(f"artifacts changed since inventory: {drift}")

    print("\nFINAL STATE VERIFICATION")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"  crash notes in test log (documented, metrics never produced by those runs): {len(crashed)}")
    print(f"  files added since inventory (not frozen): {len(new)}")
    for n in notes:
        print("  NOTE:", n)
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
