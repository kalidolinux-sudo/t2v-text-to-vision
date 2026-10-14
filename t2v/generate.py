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
    gen = T2VGenerator(tok.layer_ids, tok.intra_ids, tok.slots, valid=tok.valid, **cfg).to(device)
    gen.load_state_dict(ck["generator"])
    gen.prompt_ensemble = ck.get("prompt_ensemble", False)
    gen.arch = ck["arch"]
    gen.eval()
    return gen, tok, ck


@torch.no_grad()
def generate_flat(gen, tok, class_names, n=1, steps=100, emb=None, prior_emb=None, arch=None):
    """Weights for `n` models classifying `class_names` (in order). Returns (n, P) flat vectors."""
    K = tok.K
    if len(class_names) > K:
        raise ValueError(f"at most {K} classes")
    dev = tok.idx.device
    if emb is None:
        emb = embed_names(class_names, dev, ensemble=getattr(gen, "prompt_ensemble", False))
    full = torch.zeros(n, K, emb.shape[1], device=dev)
    full[:, :len(class_names)] = emb
    mask = torch.zeros(n, K, dtype=torch.bool, device=dev)
    mask[:, :len(class_names)] = True
    tokens = gen.sample(full, mask, steps=steps)
    flat = tok.decode(tokens, num_classes=len(class_names))
    arch = arch or getattr(gen, "arch", None)
    if arch and target.has_prior(arch):
        if prior_emb is None:
            prior_emb = embed_names(class_names, dev, ensemble=True)
        flat = target.add_head_prior(arch, flat, prior_emb)
    return flat


def embed_images(class_images, device="cpu"):
    """Tina's image prompts: condition on example photos instead of class names. CLIP puts images and
    text in one space, so the mean image embedding of each class stands in for its text embedding.
    class_images: list (one per class) of lists of uint8 (H, W, 3) arrays -> (k, 512)."""
    from .features import image_features
    embs = []
    for imgs in class_images:
        x = torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im in imgs])
        embs.append(torch.nn.functional.normalize(image_features(x, device).mean(0), dim=-1))
    return torch.stack(embs).to(device)


def load_image_dir(path, max_per_class=16):
    """<path>/<class name>/*.jpg|png -> (class names, list of image arrays per class)."""
    import numpy as np
    from PIL import Image
    names, images = [], []
    for d in sorted(os.listdir(path)):
        full = os.path.join(path, d)
        if not os.path.isdir(full):
            continue
        files = [f for f in sorted(os.listdir(full)) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if files:
            names.append(d.replace("_", " "))
            images.append([np.asarray(Image.open(os.path.join(full, f)).convert("RGB")) for f in files[:max_per_class]])
    return names, images


class Generated(torch.nn.Module):
    """Generated network + input normalisation + slicing to the requested classes (for export)."""

    def __init__(self, net, k, features=False):
        super().__init__()
        from .data import MEAN, STD
        self.net, self.k, self.features = net, k, features
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        # images: float in [0, 1], (B, 3, 32, 32); clip_head: CLIP ViT-B/32 image features, (B, 512)
        if not self.features:
            x = (x - self.mean) / self.std
        return self.net(x)[:, :self.k]

    def predict_image(self, img_uint8):
        """img_uint8: (H, W, 3) uint8 array of any size -> class probabilities."""
        x = torch.from_numpy(img_uint8).permute(2, 0, 1)[None]
        with torch.no_grad():
            if self.features:
                from .features import image_features
                inp = image_features(x)
            else:
                inp = torch.nn.functional.interpolate(x.float() / 255, size=(32, 32), mode="bicubic",
                                                      align_corners=False).clamp(0, 1)
            return self(inp).softmax(-1)[0]


def build_model(arch, flat, k):
    net = target.load_flat(target.build(arch), flat.cpu()).eval()
    return Generated(net, k, features=target.uses_features(arch)).eval()


def export(model, classes, arch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"arch": arch, "classes": classes, "state": model.net.state_dict()},
               os.path.join(out_dir, "weights.pt"))
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump(classes, f, ensure_ascii=False)
    onnx_path = os.path.join(out_dir, "model.onnx")
    try:
        dummy, name = (torch.rand(1, 512), "clip_features") if model.features else (torch.rand(1, 3, 32, 32), "image")
        torch.onnx.export(model, dummy, onnx_path, input_names=[name],
                          output_names=["logits"], dynamic_axes={name: {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False)
    except Exception as e:  # ONNX is a convenience; weights.pt is the primary artefact
        print(f"ONNX export skipped: {e}")
        onnx_path = None
    return onnx_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/t2v_tinacnn_cifar100.pt")
    p.add_argument("--prompt", help="free text in any language, or a comma-separated class list")
    p.add_argument("--image_dir", help="instead of text: one sub-folder of example photos per class (Tina's image prompts)")
    p.add_argument("--out", default="out/generated")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    dev = pick_device(a.device)
    gen, tok, ck = load_generator(a.ckpt, dev)
    emb = None
    if a.image_dir:
        classes, images = load_image_dir(a.image_dir)
        emb = embed_images(images, dev)
        print(f"classes (from images): {classes}")
    elif a.prompt:
        classes, info = parse_classes(a.prompt, tok.K)
        print(f"classes ({info['source']}): {classes}")
    else:
        raise SystemExit("give --prompt or --image_dir")
    t0 = time.time()
    flat = generate_flat(gen, tok, classes, steps=a.steps, emb=emb)[0]
    dt = time.time() - t0
    model = build_model(ck["arch"], flat, len(classes))
    onnx_path = export(model, classes, ck["arch"], a.out)
    print(f"generated {tok.num_params:,} parameters for a {ck['arch']} in {dt:.2f}s -> {a.out}"
          + (f" (+ {os.path.basename(onnx_path)})" if onnx_path else ""))


if __name__ == "__main__":
    main()
