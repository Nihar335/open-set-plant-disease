"""BioCLIP 2 local smoke test (Milestone 4 preparation). NOT an evaluation: no accuracy is computed.

Run ONLY with the isolated environment (open_clip is deliberately not installed in the project environment):
  C:\\Users\\nihar\\venvs\\agrovaidya-plantid\\Scripts\\python.exe stage10\\plant_id\\smoke_test_bioclip.py

* Offline: HF_HUB_OFFLINE=1 is forced, so nothing is downloaded; weights come from stage10_local/models/bioclip-2.
* The weights file is re-hashed and must equal the official Hugging Face LFS SHA-256 before loading.
* Inputs are SYNTHETIC JPEGs generated in memory (no project, pilot or external images are used).
* Measures: load time, GPU/CPU memory, text-encoding time for a fixed number of prompts, and per-image end-to-end latency
  (JPEG decode -> preprocess -> image encoder -> similarity to all text embeddings) on GPU (fp32, fp16 autocast) and CPU.
Writes stage10/plant_id/smoke_test_results.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "stage10_local" / "models" / "bioclip-2"
WEIGHTS = WEIGHTS_DIR / "open_clip_model.safetensors"
OUT = Path(__file__).resolve().parent / "smoke_test_results.json"
EXPECTED_SHA256 = "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef"
EXPECTED_SIZE = 1710517724
REVISION = "2957b322090f9cb17ae72c71981c7218a28d81e0"
N_PROMPTS = 60               # order of magnitude of the planned vocabulary (+ genus/family prompts later)
N_GPU, N_CPU, WARMUP = 50, 10, 5


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def synthetic_jpegs(n: int, size=(1024, 768), seed=0) -> list[bytes]:
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = (rng.random((size[1], size[0], 3)) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=90)
        out.append(buf.getvalue())
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    import psutil
    import torch
    import open_clip
    from PIL import Image

    res = {"note": "Smoke test only. Synthetic inputs; no accuracy measured; not evidence that the model works for Indian plants.",
           "model": {"repo": "imageomics/bioclip-2", "revision": REVISION, "file": WEIGHTS.name}, "errors": []}
    size = WEIGHTS.stat().st_size
    digest = sha256(WEIGHTS)
    res["model"].update(size_bytes=size, sha256=digest, sha256_matches_official=(digest == EXPECTED_SHA256 and size == EXPECTED_SIZE))
    if not res["model"]["sha256_matches_official"]:
        raise SystemExit("weights checksum mismatch: refusing to load")
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    res["environment"] = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
                          "open_clip": open_clip.__version__, "cuda_available": torch.cuda.is_available(),
                          "gpu": gpu.name if gpu else None, "gpu_total_mib": round(gpu.total_memory / 2**20) if gpu else None,
                          "cpu": platform.processor(), "cpu_logical_cores": psutil.cpu_count(), "ram_total_gib":
                          round(psutil.virtual_memory().total / 2**30, 1), "venv": sys.prefix}
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(f"local-dir:{WEIGHTS_DIR}")
    tokenizer = open_clip.get_tokenizer(f"local-dir:{WEIGHTS_DIR}")
    model.eval()
    res["load"] = {"seconds_cpu": round(time.perf_counter() - t, 2), "rss_increase_mib": round((proc.memory_info().rss - rss0) / 2**20),
                   "n_parameters": sum(p.numel() for p in model.parameters())}
    prompts = [f"a photo of Plantae Tracheophyta Magnoliopsida test taxon {i}" for i in range(N_PROMPTS)]
    jpegs = synthetic_jpegs(N_GPU + WARMUP, seed=1)

    def run(device, n, autocast):
        m = model.to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast):
            t0 = time.perf_counter()
            tf = m.encode_text(tokenizer(prompts).to(device))
            tf = tf / tf.norm(dim=-1, keepdim=True)
            if device == "cuda":
                torch.cuda.synchronize()
            text_s = time.perf_counter() - t0
            lat = []
            for i, b in enumerate(jpegs[: n + WARMUP]):
                t1 = time.perf_counter()
                x = preprocess(Image.open(io.BytesIO(b)).convert("RGB")).unsqueeze(0).to(device)
                f = m.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                (100.0 * f @ tf.T).softmax(dim=-1).cpu()
                if device == "cuda":
                    torch.cuda.synchronize()
                if i >= WARMUP:
                    lat.append(time.perf_counter() - t1)
        return {"n_images": len(lat), "warmup_images": WARMUP, "text_encode_seconds_for_prompts": round(text_s, 3), "n_prompts": N_PROMPTS,
                "latency_ms_p50": round(1000 * statistics.median(lat), 1), "latency_ms_p95": round(1000 * pct(lat, 0.95), 1),
                "latency_ms_max": round(1000 * max(lat), 1)}

    if torch.cuda.is_available():
        for name, ac in (("gpu_fp32", False), ("gpu_fp16_autocast", True)):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                r = run("cuda", N_GPU, ac)
                r["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
                r["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
                res[name] = r
            except Exception as e:  # report, do not hide
                res["errors"].append(f"{name}: {type(e).__name__}: {e}")
        model.to("cpu")
        torch.cuda.empty_cache()
    try:
        rss1 = proc.memory_info().rss
        res["cpu_fp32"] = run("cpu", N_CPU, False)
        res["cpu_fp32"]["rss_mib_after"] = round(proc.memory_info().rss / 2**20)
        res["cpu_fp32"]["rss_increase_mib"] = round((proc.memory_info().rss - rss1) / 2**20)
    except Exception as e:
        res["errors"].append(f"cpu_fp32: {type(e).__name__}: {e}")
    res["latency_definition"] = ("per image, end to end: JPEG decode (synthetic 1024x768) + preprocessing + image encoder + "
                                 "cosine similarity/softmax against precomputed text embeddings; batch size 1; after warm-up")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
