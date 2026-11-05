"""Level-4 duplicate audit: CNN-embedding near-duplicate search.

Embeds every readable image with an ImageNet-pretrained ResNet-50 (torchvision weights,
never trained on PlantVillage/PlantDoc -> no information from our data enters it).
Each embedding is the mean over the 8 flips/rotations, so mirrored/rotated copies score ~1.
For every image the top-k cosine neighbours are found on the GPU; pairs above
`near_duplicates.report_cosine` are written for review.

These pairs are *candidates*: nothing is deleted. Whether high-cosine pairs are merged into
the same split group is decided by `near_duplicates.merge_cosine` in config.yaml, after
the visual audit documented in DATASET_CARD.md.

Outputs:
  data/processed/embeddings_resnet50_d4.npy          float16 (N, 2048), row order = inventory rows used
  data/processed/embedding_index.csv
  data/processed/embedding_pairs.csv                  method/score/paths/classes for every flagged pair
  results/metrics/near_duplicates_embedding.json     cosine statistics + counts by category

Usage:  python scripts/find_near_duplicates.py [--recompute]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.duplicates import describe_pairs  # noqa: E402
from src.utils.config import get_device, load_config, resolve, set_seed  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ImageList(Dataset):
    def __init__(self, paths: list[str], size: int):
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(ROOT / self.paths[i]) as im:
            im = im.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)


def d4_views(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) -> (8B,3,H,W): all flips/rotations."""
    views = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        views += [r, torch.flip(r, dims=(3,))]
    return torch.cat(views, 0)


@torch.no_grad()
def embed(paths: list[str], cfg: dict, device) -> np.ndarray:
    import torchvision

    nd = cfg["near_duplicates"]
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(ImageList(paths, nd["image_size"]), batch_size=nd["batch_size"],
                        num_workers=nd["num_workers"], pin_memory=device.type == "cuda")
    out = []
    mean, std = MEAN.to(device), STD.to(device)
    for x in tqdm(loader, desc="  embedding"):
        x = (x.to(device, non_blocking=True).float() / 255 - mean) / std
        b = x.shape[0]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model(d4_views(x)).float()
        f = f.view(8, b, -1).mean(0)
        out.append(torch.nn.functional.normalize(f, dim=1).half().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def topk_pairs(emb: np.ndarray, k: int, min_cos: float, device, chunk: int = 4096) -> pd.DataFrame:
    E = torch.from_numpy(emb).to(device, torch.float16 if device.type == "cuda" else torch.float32)
    rows = []
    for s in range(0, E.shape[0], chunk):
        sim = (E[s:s + chunk] @ E.T).float()
        idx = torch.arange(s, min(s + chunk, E.shape[0]), device=device)
        sim[torch.arange(len(idx)), idx] = -1                     # drop self-match
        val, nbr = sim.topk(k, dim=1)
        keep = val >= min_cos
        i = idx.unsqueeze(1).expand_as(nbr)[keep]
        rows.append(pd.DataFrame({"i": i.cpu().numpy(), "j": nbr[keep].cpu().numpy(),
                                  "score": val[keep].cpu().numpy().round(4)}))
    p = pd.concat(rows, ignore_index=True)
    p[["i", "j"]] = np.sort(p[["i", "j"]].to_numpy(), axis=1)   # undirected, dedupe
    return p.drop_duplicates(["i", "j"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    nd = cfg["near_duplicates"]
    proc, metrics = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["metrics"])
    device = get_device()
    print(f"device: {device}")

    inv = pd.read_csv(proc / "inventory.csv", low_memory=False, keep_default_na=False)
    inv["ok"] = inv["ok"].astype(str).str.lower().eq("true")
    ok_idx = np.flatnonzero(inv["ok"].to_numpy())
    paths = inv.loc[ok_idx, "rel_path"].tolist()

    emb_file, idx_file = proc / "embeddings_resnet50_d4.npy", proc / "embedding_index.csv"
    if emb_file.exists() and idx_file.exists() and not args.recompute and \
            pd.read_csv(idx_file)["rel_path"].tolist() == paths:
        emb = np.load(emb_file)
        print(f"  [cached] embeddings {emb.shape}")
    else:
        emb = embed(paths, cfg, device)
        np.save(emb_file, emb)
        pd.DataFrame({"rel_path": paths}).to_csv(idx_file, index=False)

    p = topk_pairs(emb, nd["top_k"], nd["report_cosine"], device)
    p["i"], p["j"] = ok_idx[p["i"]], ok_idx[p["j"]]
    pairs = describe_pairs(p, inv, "L4_resnet50_D4_cosine", "candidate")
    pairs.to_csv(proc / "embedding_pairs.csv", index=False)

    # nearest-neighbour cosine for PlantVillage, split by whether the NN is the same physical leaf
    pv = np.flatnonzero((inv.loc[ok_idx, "dataset"] == "plantvillage").to_numpy())
    E = torch.from_numpy(emb[pv]).to(device, torch.float16 if device.type == "cuda" else torch.float32)
    lg = inv.loc[ok_idx[pv], "leaf_group"].to_numpy()
    nn_cos, nn_same = [], []
    for s in range(0, len(pv), 4096):
        sim = (E[s:s + 4096] @ E.T).float()
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -1
        v, j = sim.max(1)
        nn_cos.append(v.cpu().numpy()); nn_same.append(lg[j.cpu().numpy()] == lg[s:s + sim.shape[0]])
    nn_cos, nn_same = np.concatenate(nn_cos), np.concatenate(nn_same)
    q = [0.5, 0.9, 0.99, 0.999]
    stats = {
        "model": "torchvision resnet50 IMAGENET1K_V2, global-avg-pool 2048-d, mean over 8 D4 views, L2-normalised",
        "report_cosine": nd["report_cosine"], "top_k": nd["top_k"],
        "plantvillage_nearest_neighbour_cosine": {
            "nn_is_same_leaf_group": {"n": int(nn_same.sum()), **{f"q{x}": float(np.quantile(nn_cos[nn_same], x)) for x in q}},
            "nn_is_other_leaf_group": {"n": int((~nn_same).sum()), **{f"q{x}": float(np.quantile(nn_cos[~nn_same], x)) for x in q}},
        },
        "pairs_by_category": {c: {"pairs": int(len(g)), "cos>=0.98": int((g["score"] >= 0.98).sum()),
                                  "cos>=0.99": int((g["score"] >= 0.99).sum())}
                              for c, g in pairs.groupby("category")},
    }
    (metrics / "near_duplicates_embedding.json").write_text(json.dumps(stats, indent=2))
    np.save(proc / "pv_nn_cosine.npy", np.stack([nn_cos, nn_same.astype(np.float32)]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
