"""CIFAR-100 (loaded from the Hugging Face parquet mirror) and Tina-style task sampling."""
import io
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HF_REPO = "uoft-cs/cifar100"
MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


def _load_split(split, cache_dir):
    path = os.path.join(cache_dir, f"cifar100_{split}.pt")
    if os.path.exists(path):
        return torch.load(path)
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    p = hf_hub_download(HF_REPO, f"cifar100/{split}-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(p)
    meta = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    # The mirror's metadata has a typo ("cra"); everything else matches the official names.
    names = ["crab" if n == "cra" else n for n in meta["fine_label"]["names"]]
    imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                     for d in table.column("img").to_pylist()])
    out = {
        "x": torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous(),  # uint8 (N,3,32,32)
        "y": torch.tensor(table.column("fine_label").to_pylist()),
        "coarse": torch.tensor(table.column("coarse_label").to_pylist()),
        "names": names,
    }
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(out, path)
    return out


class Cifar100:
    def __init__(self, cache_dir="data", device="cpu"):
        tr, te = _load_split("train", cache_dir), _load_split("test", cache_dir)
        self.names = tr["names"]
        self.train_x, self.train_y = tr["x"].to(device), tr["y"].to(device)
        self.test_x, self.test_y = te["x"].to(device), te["y"].to(device)
        coarse = torch.zeros(100, dtype=torch.long)
        coarse[tr["y"]] = tr["coarse"]
        self.coarse = coarse.tolist()

    def subset(self, classes, train=True):
        """Images of `classes`, relabelled 0..len(classes)-1 in the given order."""
        x, y = (self.train_x, self.train_y) if train else (self.test_x, self.test_y)
        cls = torch.tensor(classes, device=y.device)
        mask = (y.unsqueeze(1) == cls).any(1)
        lut = torch.full((100,), -1, dtype=torch.long, device=y.device)
        lut[cls] = torch.arange(len(classes), device=y.device)
        return x[mask], lut[y[mask]]


def normalize(x_uint8):
    x = x_uint8.float().div(255)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x):
    """Random horizontal flip per image + one random 32x32 crop of a 4px-padded batch."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,)).tolist()
    return x[:, :, i:i + 32, j:j + 32]


def split_classes(num_holdout, seed=0):
    """Fixed split into classes seen by the zoo and classes held out for unseen-class evaluation."""
    rng = random.Random(seed)
    classes = list(range(100))
    rng.shuffle(classes)
    return sorted(classes[num_holdout:]), sorted(classes[:num_holdout])


def sample_task(rng, pool, coarse, min_k=10, max_k=10):
    """Tina's 3:2:1 mix of random / partly-same-superclass / two-superclass tasks."""
    k = rng.randint(min_k, max_k)
    by_super = defaultdict(list)
    for c in pool:
        by_super[coarse[c]].append(c)
    supers = [s for s, cs in by_super.items() if len(cs) >= 2]
    mode = rng.choices(["random", "partial", "two_super"], weights=[3, 2, 1])[0]
    chosen = []
    if mode == "two_super" and len(supers) >= 2:
        for s in rng.sample(supers, 2):
            chosen += rng.sample(by_super[s], min(len(by_super[s]), k // 2))
    elif mode == "partial" and supers:
        s = rng.choice(supers)
        chosen += rng.sample(by_super[s], min(len(by_super[s]), rng.randint(2, max(2, k // 2))))
    rest = [c for c in pool if c not in chosen]
    chosen += rng.sample(rest, k - len(chosen))
    rng.shuffle(chosen)
    return chosen[:k]
