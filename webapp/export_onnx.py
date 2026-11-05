"""Export the LeafScan model for the browser version.

Run:  python -P webapp/export_onnx.py <output.onnx>        (needs: onnx)

Outputs: disease logits (1,43), symptom logits (1,25), class activation maps (1,43,7,7).

Grad-CAM needs no gradients for this network: a class logit is  w_c . avgpool(fmap) + b_c,  so d logit_c / d fmap is
w_c / 49 everywhere, and Grad-CAM's (mean gradient * fmap) summed over channels equals the 1x1 convolution of fmap with
w_c, up to a scale that the normalisation in the app removes. The maps are therefore part of the exported graph.

Weights are stored as float16 and cast to float32 when the model loads, which halves the download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.model import IMG_SIZE, LeafNet  # noqa: E402


class Exported(nn.Module):
    def __init__(self, net: LeafNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        fmap = self.net.features(x)
        z = torch.flatten(self.net.pool(fmap), 1)
        w = self.net.disease_head.weight[:, :, None, None]
        return self.net.disease_head(z), self.net.symptom_head(z), F.conv2d(fmap, w)


def to_float16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Store large float32 initializers as float16, with a Cast back to float32 at the start of the graph."""
    graph = model.graph
    casts, keep = [], []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type != TensorProto.FLOAT or arr.size < 16:
            keep.append(init)
            continue
        half = init.name + "__float16"
        keep.append(numpy_helper.from_array(arr.astype(np.float16), half))
        casts.append(helper.make_node("Cast", [half], [init.name], to=TensorProto.FLOAT, name="cast_" + init.name))
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def main(out: Path) -> None:
    ckpt = torch.load(ROOT / "models/stage11/best.pt", map_location="cpu", weights_only=False)
    net = LeafNet(len(ckpt["classes"]), len(ckpt["symptoms"]), pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    raw = out.with_suffix(".float32.onnx")
    torch.onnx.export(Exported(net), torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), str(raw), opset_version=17,
                      input_names=["image"], output_names=["logits", "symptoms", "cams"], do_constant_folding=True,
                      dynamo=False)
    model = onnx.load(str(raw))
    onnx.save(to_float16(model), str(out))
    onnx.checker.check_model(str(out))
    print(f"float32 {raw.stat().st_size / 1e6:.1f} MB, float16 {out.stat().st_size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
