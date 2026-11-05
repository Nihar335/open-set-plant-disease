"""Seeding, determinism and environment capture."""
from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    # warn_only: a few CUDA kernels (e.g. some backward ops) have no deterministic implementation;
    # small run-to-run GPU differences are therefore still possible.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def environment(seed: int, device: torch.device) -> dict:
    import torchvision

    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "torchvision": torchvision.__version__,
           "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "device": str(device), "seed": seed,
           "cudnn_deterministic": torch.backends.cudnn.deterministic,
           "cudnn_benchmark": torch.backends.cudnn.benchmark, "numpy": np.__version__}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env.update({"gpu": p.name, "gpu_memory_gb": round(p.total_memory / 1024**3, 2),
                    "gpu_capability": f"{p.major}.{p.minor}"})
    env["note"] = ("Seeds are fixed and cuDNN is deterministic, but some CUDA kernels have no deterministic "
                   "implementation, so reruns may differ slightly.")
    return env
