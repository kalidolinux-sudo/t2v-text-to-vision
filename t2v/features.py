"""Frozen CLIP image features for CIFAR-100, computed once (python -m t2v.features).

They back the `clip_head` target: the generator writes a classifier head on top of a frozen CLIP
ViT-B/32, which is Tina's ViT setting and works on real photos instead of 32x32 thumbnails.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from .data import _load_split
from .text_embed import _load

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_pixels(x_uint8, size=224):
    """uint8 (N,3,H,W) -> CLIP-normalised float (N,3,224,224)."""
    x = F.interpolate(x_uint8.float() / 255, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)


@torch.no_grad()
def image_features(x_uint8, device="cpu", bs=256):
    model, _ = _load(device)
    out = []
    for i in range(0, len(x_uint8), bs):
        f = model.get_image_features(pixel_values=clip_pixels(x_uint8[i:i + bs].to(device)))
        if not torch.is_tensor(f):
            f = f.pooler_output if getattr(f, "pooler_output", None) is not None else f[0]
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bs", type=int, default=256)
    a = p.parse_args()
    for split in ("test", "train"):
        path = os.path.join(a.data, f"clip_feats_{split}.pt")
        if os.path.exists(path):
            print(f"{path} exists")
            continue
        x = _load_split(split, a.data)["x"]
        t0, feats = time.time(), []
        for i in range(0, len(x), 2000):
            feats.append(image_features(x[i:i + 2000], a.device, a.bs))
            print(f"{split}: {i + len(feats[-1])}/{len(x)}  {time.time() - t0:.0f}s", flush=True)
        torch.save(torch.cat(feats).half(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
