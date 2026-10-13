"""Text in, vision model out: `python -m t2v.generate --prompt "cats, dogs and horses"`."""
import argparse
import json
import os
import time

import torch

from . import target
from .generator import T2VGenerator
from .llm import parse_classes
from .text_embed import embed_names
from .tokenizer import ParamTokenizer
from .zoo import pick_device


def load_generator(path, device):
    ck = torch.load(path, map_location=device)
    tok = ParamTokenizer.from_state(ck["tokenizer"]).to(device)
    cfg = {"pred": "eps", "clip": 3.0, **ck["config"]}  # defaults for checkpoints saved before these options
    gen = T2VGenerator(tok.layer_ids, tok.intra_ids, tok.slots, **cfg).to(device)
    gen.load_state_dict(ck["generator"])
    gen.eval()
    return gen, tok, ck


@torch.no_grad()
def generate_flat(gen, tok, class_names, n=1, steps=100, emb=None):
    """Weights for `n` models classifying `class_names` (in order). Returns (n, P) flat vectors."""
    K = tok.K
    if len(class_names) > K:
        raise ValueError(f"at most {K} classes")
    dev = tok.idx.device
    if emb is None:
        emb = embed_names(class_names, dev)
    full = torch.zeros(n, K, emb.shape[1], device=dev)
    full[:, :len(class_names)] = emb
    mask = torch.zeros(n, K, dtype=torch.bool, device=dev)
    mask[:, :len(class_names)] = True
    tokens = gen.sample(full, mask, steps=steps)
    return tok.decode(tokens, num_classes=len(class_names))


class Generated(torch.nn.Module):
    """Generated network + input normalisation + slicing to the requested classes (for export)."""

    def __init__(self, net, k):
        super().__init__()
        from .data import MEAN, STD
        self.net, self.k = net, k
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):  # x: float images in [0, 1], (B, 3, 32, 32)
        return self.net((x - self.mean) / self.std)[:, :self.k]


def build_model(arch, flat, k):
    return Generated(target.load_flat(target.build(arch), flat.cpu()).eval(), k).eval()


def export(model, classes, arch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"arch": arch, "classes": classes, "state": model.net.state_dict()},
               os.path.join(out_dir, "weights.pt"))
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump(classes, f, ensure_ascii=False)
    onnx_path = os.path.join(out_dir, "model.onnx")
    try:
        torch.onnx.export(model, torch.rand(1, 3, 32, 32), onnx_path, input_names=["image"],
                          output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False)
    except Exception as e:  # ONNX is a convenience; weights.pt is the primary artefact
        print(f"ONNX export skipped: {e}")
        onnx_path = None
    return onnx_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/generator.pt")
    p.add_argument("--prompt", required=True, help="free text in any language, or a comma-separated class list")
    p.add_argument("--out", default="out/generated")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    dev = pick_device(a.device)
    gen, tok, ck = load_generator(a.ckpt, dev)
    classes, info = parse_classes(a.prompt, tok.K)
    print(f"classes ({info['source']}): {classes}")
    t0 = time.time()
    flat = generate_flat(gen, tok, classes, steps=a.steps)[0]
    dt = time.time() - t0
    model = build_model(ck["arch"], flat, len(classes))
    onnx_path = export(model, classes, ck["arch"], a.out)
    print(f"generated {tok.num_params:,} parameters for a {ck['arch']} in {dt:.2f}s -> {a.out}"
          + (f" (+ {os.path.basename(onnx_path)})" if onnx_path else ""))


if __name__ == "__main__":
    main()
