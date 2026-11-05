"""Exploratory BioCLIP 2 pilot evaluation (protocol v1 + amendment 1). Run ONLY with the isolated environment:

  C:\\Users\\nihar\\venvs\\agrovaidya-plantid\\Scripts\\python.exe -m stage10.plant_id.pilot.evaluate_pilot --phase dev
  C:\\Users\\nihar\\venvs\\agrovaidya-plantid\\Scripts\\python.exe -m stage10.plant_id.pilot.evaluate_pilot --phase sealed

Phase dev   : P1/P2 DEVELOPMENT images only. Template by the frozen rule; abstention tau by the frozen rule; CPU latency.
              Writes protocol_v1_operating_point.json (refuses to overwrite). No sealed image is read.
Phase sealed: requires SEALED_SCORING_APPROVAL.json bound to the current protocol/amendment/seal hashes and the operating
              point; scores every P1/P2 sealed image ONCE; writes results/. Nothing is tuned; P1 and P2 reported separately.
EXPLORATORY ONLY: cannot establish suitability for Indian agriculture or that the full gates are met.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from stage10.plant_id.pilot import common as C  # noqa: E402

H = C.HERE
M = H / "manifests"
OUT = H / "results"
OP = H / "protocol_v1_operating_point.json"
WEIGHTS_DIR = C.ROOT / "stage10_local" / "models" / "bioclip-2"
WEIGHTS_SHA = "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(name):
    with open(M / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_model():
    import open_clip
    import torch
    w = WEIGHTS_DIR / "open_clip_model.safetensors"
    h = hashlib.sha256()
    with open(w, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != WEIGHTS_SHA:
        raise SystemExit("weights checksum mismatch")
    model, _, pre = open_clip.create_model_and_transforms(f"local-dir:{WEIGHTS_DIR}")
    tok = open_clip.get_tokenizer(f"local-dir:{WEIGHTS_DIR}")
    model.eval()
    return model, pre, tok, torch


def prompts(vocab, template):
    proto = json.loads((H / "protocol_v1.json").read_text(encoding="utf-8"))["prompt_templates"][template]
    out = []
    for t in vocab["taxa"]:
        out.append(proto.format(kingdom=t["kingdom"], phylum=t["phylum"], **{"class": t["class"]}, order=t["order"], family=t["family"],
                                genus=t["genus"], species_epithet=t["scientific_name"].split()[1], common_name=t["common_name_en"],
                                scientific_name=t["scientific_name"]))
    return out


class Scorer:
    def __init__(self, template, device):
        self.model, self.pre, self.tok, self.torch = load_model()
        self.device = device
        self.model.to(device)
        self.vocab = json.loads((H / "vocabulary_v1.json").read_text(encoding="utf-8"))
        with self.torch.no_grad():
            tf = self.model.encode_text(self.tok(prompts(self.vocab, template)).to(device))
            self.tf = tf / tf.norm(dim=-1, keepdim=True)
            self.scale = self.model.logit_scale.exp()

    def score(self, path: Path):
        """End-to-end: file read + decode + preprocess + image encoder + softmax over the 40 vocabulary prompts."""
        from PIL import Image
        torch = self.torch
        t0 = time.perf_counter()
        im = Image.open(io.BytesIO(path.read_bytes())).convert("RGB")
        x = self.pre(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (self.scale * f @ self.tf.T).softmax(dim=-1)[0].float().cpu()
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        top = torch.topk(p, 5)
        return [(self.vocab["taxa"][i]["scientific_name"], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())], ms


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# ---------------------------------------------------------------------------------------------------------- dev phase
def phase_dev():
    if OP.exists():
        raise SystemExit("operating point already frozen; refusing to overwrite")
    proto = json.loads((H / "protocol_v1.json").read_text(encoding="utf-8"))
    dev = read("p1p2_dev_manifest.csv")
    p1, p2 = [r for r in dev if r["block"] == "P1"], [r for r in dev if r["block"] == "P2"]
    template = "T1"                          # frozen rule: < 30 in-vocabulary dev images -> T1 (no template comparison is run)
    if len(p1) >= 30:
        raise SystemExit("P1 dev >= 30: the template comparison branch of the rule is not implemented in this pilot")
    sc = Scorer(template, "cuda")
    dev_scores = {}
    for r in dev:
        top, _ = sc.score(C.ROOT / r["stored_path"])
        dev_scores[r["record_id"]] = top
    p2_max = sorted(dev_scores[r["record_id"]][0][1] for r in p2)
    # smallest tau with P2-dev false acceptance <= 0.10 (answer iff top-1 score >= tau)
    cands = sorted({0.0} | {s for s in p2_max} | {s + 1e-9 for s in p2_max})
    tau = next((t for t in cands if sum(s >= t for s in p2_max) / len(p2_max) <= proto_fa(proto)), 1.0)
    fa = sum(s >= tau for s in p2_max) / len(p2_max)
    p1_cov = sum(dev_scores[r["record_id"]][0][1] >= tau for r in p1) / len(p1)
    # CPU latency on DEVELOPMENT images only
    cpu = Scorer(template, "cpu")
    cpu_ms = [cpu.score(C.ROOT / r["stored_path"])[1] for r in dev]
    op = {"created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "template": template,
          "template_rule_applied": f"P1 dev has {len(p1)} in-vocabulary images (< 30) -> T1; no template comparison performed",
          "tau": tau, "tau_rule": proto["abstention"]["tau_selection"],
          "dev": {"p1_images": len(p1), "p2_images": len(p2), "p2_false_acceptance_at_tau": fa, "p2_false_acceptance_resolution": 1 / len(p2),
                  "p1_coverage_at_tau": p1_cov},
          "cpu_latency_ms_dev": {"n": len(cpu_ms), "p50": round(statistics.median(cpu_ms), 1), "p95": round(pct(cpu_ms, 0.95), 1)},
          "note": "Frozen BEFORE any sealed image is read. Dev-only. Never changed afterwards."}
    OP.write_text(json.dumps(op, indent=1), encoding="utf-8")
    print(json.dumps(op, indent=1))
    print("operating point sha256:", sha(OP))


def proto_fa(proto):
    return 0.10                               # protocol v1 abstention rule: false acceptance on P2 dev <= 0.10


# ---------------------------------------------------------------------------------------------------------- sealed phase
def bootstrap(rows, stat, clusters, n=2000, seed=0):
    rng = random.Random(seed)
    keys = sorted({r[clusters] for r in rows})
    by = {k: [r for r in rows if r[clusters] == k] for k in keys}
    vals = []
    for _ in range(n):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)] if vals else None


def phase_sealed():
    approval = C.assert_sealed_access_approved()
    lock = json.loads((H / "protocol_lock.json").read_text(encoding="utf-8"))
    for k, f in (("protocol_sha256", "protocol_v1.json"), ("amendment_sha256", "protocol_v1_amendment_1.json"),
                 ("vocabulary_sha256", "vocabulary_v1.json")):
        if approval[k] != sha(H / f) or lock["files"][f] != sha(H / f):
            raise SystemExit(f"{f} changed since approval: refusing to score")
    if approval["seal_sha256"] != sha(H / "SEAL.json"):
        raise SystemExit("seal changed since approval: refusing to score")
    if not OP.exists():
        raise SystemExit("operating point not frozen (run --phase dev first)")
    if OUT.exists():
        raise SystemExit("sealed results already exist: the sealed set is scored once")
    op = json.loads(OP.read_text(encoding="utf-8"))
    vocab = json.loads((H / "vocabulary_v1.json").read_text(encoding="utf-8"))
    tax = {t["scientific_name"]: t for t in vocab["taxa"]}
    groups = vocab["look_alike_groups"]
    caveats = {r["stored_sha256"]: f"{r['flag']}: {r['note']}" for r in read("visual_review.csv") if r["owner_decision"].startswith("keep")}
    sealed = read("p1p2_sealed_manifest.csv")
    import torch
    sc = Scorer(op["template"], "cuda")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):                                    # GPU warm-up on a synthetic image (not a sealed image)
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (512, 384), (90, 140, 60)).save(buf, format="JPEG")
        tmp = OUT.parent / "_warmup.jpg"
        tmp.write_bytes(buf.getvalue())
        sc.score(tmp)
        tmp.unlink()
    recs = []
    for r in sealed:
        top, ms = sc.score(C.ROOT / r["stored_path"])
        t1, s1 = top[0]
        truth = r["taxon"]
        rec = {"block": r["block"], "taxon": truth, "record_id": r["record_id"], "photographer_key": r["photographer_key"],
               "observation_url": r["observation_url"], "stored_sha256": r["stored_sha256"], "caveat": caveats.get(r["stored_sha256"], ""),
               "top1": t1, "top1_score": round(s1, 6), "top5": "|".join(f"{n}:{s:.4f}" for n, s in top), "answered": s1 >= op["tau"],
               "latency_ms_gpu_fp32": round(ms, 2)}
        if r["block"] == "P1":
            rec.update(top1_correct=t1 == truth, top5_correct=truth in [n for n, _ in top],
                       genus_correct=tax[t1]["genus"] == tax[truth]["genus"], family_correct=tax[t1]["family"] == tax[truth]["family"])
        recs.append(rec)
    peak = round(torch.cuda.max_memory_allocated() / 2**20)
    OUT.mkdir()
    with open(OUT / "pilot_sealed_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in recs for k in r}))
        w.writeheader()
        w.writerows(recs)
    p1 = [r for r in recs if r["block"] == "P1"]
    p2 = [r for r in recs if r["block"] == "P2"]

    def rate(rows, key):
        return sum(bool(r[key]) for r in rows) / len(rows) if rows else None
    acc_tau = lambda rows: (sum(r["answered"] and r["top1_correct"] for r in rows) / len(rows)) if rows else None  # noqa: E731
    cov = lambda rows: rate(rows, "answered")  # noqa: E731
    sel = lambda rows: (sum(r["top1_correct"] for r in rows if r["answered"]) / max(1, sum(r["answered"] for r in rows))) if any(r["answered"] for r in rows) else None  # noqa: E731
    # risk-coverage / AURC over the top-1 score
    order = sorted(p1, key=lambda r: -r["top1_score"])
    risks, err = [], 0
    for i, r in enumerate(order, 1):
        err += not r["top1_correct"]
        risks.append(err / i)
    aurc = sum(risks) / len(risks)
    per_taxon = {}
    for t in sorted({r["taxon"] for r in recs}):
        rows = [r for r in recs if r["taxon"] == t]
        if rows[0]["block"] == "P1":
            d = {"n": len(rows), "top1_correct": sum(r["top1_correct"] for r in rows), "top5_correct": sum(r["top5_correct"] for r in rows),
                 "answered": sum(r["answered"] for r in rows), "answered_and_correct": sum(r["answered"] and r["top1_correct"] for r in rows)}
        else:
            d = {"n": len(rows), "wrongly_accepted": sum(r["answered"] for r in rows)}
        d["percentages_shown"] = len(rows) >= 10
        if d["percentages_shown"]:
            d.update({k + "_pct": round(100 * v / len(rows), 1) for k, v in list(d.items()) if k not in ("n", "percentages_shown")})
        per_taxon[t] = d
    look = {}
    for g, members in groups.items():
        rows = [r for r in p1 if r["taxon"] in members]
        if rows:
            look[g] = {"n": len(rows), "confused_within_group": sum(r["top1"] in members and r["top1"] != r["taxon"] for r in rows),
                       "correct": sum(r["top1_correct"] for r in rows)}
    from collections import Counter
    lat = [r["latency_ms_gpu_fp32"] for r in recs]
    res = {
        "status": "EXPLORATORY PILOT RESULTS - proposed for owner review",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "protocol": {"protocol_sha256": approval["protocol_sha256"], "amendment_sha256": approval["amendment_sha256"],
                     "operating_point_sha256": sha(OP), "template": op["template"], "tau": op["tau"]},
        "P1_known_vocabulary_identification": {
            "n_images": len(p1), "n_photographers": len({r["photographer_key"] for r in p1}),
            "top1_accuracy_at_tau_abstentions_as_errors": acc_tau(p1), "coverage_at_tau": cov(p1), "selective_accuracy_at_tau": sel(p1),
            "top1_accuracy_no_abstention_supplementary": rate(p1, "top1_correct"), "top5_accuracy": rate(p1, "top5_correct"),
            "genus_accuracy_of_top1": rate(p1, "genus_correct"), "family_accuracy_of_top1": rate(p1, "family_correct"), "aurc": round(aurc, 4),
            "ci95_photographer_bootstrap": {"top1_at_tau": bootstrap(p1, acc_tau, "photographer_key"), "coverage": bootstrap(p1, cov, "photographer_key"),
                                            "selective_accuracy": bootstrap(p1, sel, "photographer_key"),
                                            "top1_no_abstention": bootstrap(p1, lambda x: rate(x, "top1_correct"), "photographer_key"),
                                            "top5": bootstrap(p1, lambda x: rate(x, "top5_correct"), "photographer_key")},
            "most_frequent_wrong_top1": Counter(r["top1"] for r in p1 if not r["top1_correct"]).most_common(8)},
        "P2_rejection_of_unsupported_plants": {
            "n_images": len(p2), "n_photographers": len({r["photographer_key"] for r in p2}),
            "false_acceptance_rate_at_tau": rate(p2, "answered"), "correctly_rejected_rate_at_tau": 1 - rate(p2, "answered"),
            "ci95_photographer_bootstrap": {"false_acceptance": bootstrap(p2, lambda x: rate(x, "answered"), "photographer_key")},
            "most_frequent_wrong_labels_when_accepted": Counter(r["top1"] for r in p2 if r["answered"]).most_common(8),
            "most_frequent_top1_overall": Counter(r["top1"] for r in p2).most_common(8)},
        "per_taxon": per_taxon, "look_alike_groups": look,
        "performance": {"gpu": "NVIDIA GeForce RTX 4050 Laptop GPU, fp32, batch 1, end-to-end (file read to scores)",
                        "gpu_latency_ms": {"n": len(lat), "p50": round(statistics.median(lat), 1), "p95": round(pct(lat, 0.95), 1), "max": round(max(lat), 1)},
                        "gpu_peak_allocated_mib": peak, "cpu_latency_ms_dev_images": op["cpu_latency_ms_dev"]},
        "provisional_gates_reference_only": {"note": "Gates apply to the FULL evaluation; this pilot cannot establish them.",
                                             "top1_min_0.80_at_coverage_min_0.70": None, "unsupported_fa_max_0.10": None, "p95_latency_max_2000ms": None},
        "caveated_images_scored": sum(bool(r["caveat"]) for r in recs),
        "not_scored": {"quarantined_pending_expert": 2, "excluded_by_review": 3, "P3": "dropped (deferred)"},
    }
    (OUT / "pilot_sealed_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "sealed"], required=True)
    a = ap.parse_args()
    if "agrovaidya-plantid" not in sys.prefix:
        raise SystemExit("run with the isolated environment C:\\Users\\nihar\\venvs\\agrovaidya-plantid")
    phase_dev() if a.phase == "dev" else phase_sealed()


if __name__ == "__main__":
    main()
