"""Datasets (CIFAR-100, ImageNet-1k 32x32; both from Hugging Face parquet mirrors) and task sampling."""
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
IMAGENET_REPO = "benjamin-paine/imagenet-1k-32x32"
MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


def _decode(images):
    return torch.from_numpy(np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                                      for d in images])).permute(0, 3, 1, 2).contiguous()


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
    out = {
        "x": _decode(table.column("img").to_pylist()),  # uint8 (N,3,32,32)
        "y": torch.tensor(table.column("fine_label").to_pylist()),
        "coarse": torch.tensor(table.column("coarse_label").to_pylist()),
        "names": names,
    }
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(out, path)
    return out


def imagenet_names(full_names):
    """Short, unique class names: the first synonym, plus the second one when the first is ambiguous."""
    short = [n.split(",")[0].strip() for n in full_names]
    dup = {s for s in short if short.count(s) > 1}
    out = []
    for s, n in zip(short, full_names):
        parts = [p.strip() for p in n.split(",")]
        out.append(f"{s} ({parts[1]})" if s in dup and len(parts) > 1 else s)
    seen = defaultdict(int)  # still identical (no second synonym): number them
    for i, s in enumerate(out):
        seen[s] += 1
        if out.count(s) > 1 and seen[s] > 1:
            out[i] = f"{s} {seen[s]}"
    return out


def _load_imagenet(split, cache_dir, per_class):
    """ImageNet-1k at 32x32. train: the first `per_class` images of each class (the train shards);
    test: the labelled validation split (50 per class), which the zoo never sees."""
    tag = f"train{per_class}" if split == "train" else "val"
    path = os.path.join(cache_dir, f"imagenet32_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path)
    if split == "train":  # a larger cached download holds the first `per_class` images of every class too
        import glob
        sizes = sorted(int(os.path.basename(f)[len("imagenet32_train"):-3]) for f in glob.glob(os.path.join(cache_dir, "imagenet32_train*.pt"))
                       if os.path.basename(f)[len("imagenet32_train"):-3].isdigit())
        bigger = [n for n in sizes if n > per_class]
        if bigger:
            big = torch.load(os.path.join(cache_dir, f"imagenet32_train{bigger[0]}.pt"))
            y = big["y"]
            order = torch.argsort(y, stable=True)
            first = torch.zeros(len(y), dtype=torch.long)
            counts = torch.bincount(y, minlength=1000)
            starts = torch.cumsum(counts, 0) - counts
            first[order] = torch.arange(len(y)) - starts[y[order]]   # rank of each image within its class
            keep = first < per_class
            return {"x": big["x"][keep], "y": y[keep], "names": big["names"]}
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    files = ([f"data/train-0000{i}-of-00003.parquet" for i in range(3)] if split == "train"
             else ["data/validation-00000-of-00001.parquet"])
    xs, ys, names, count = [], [], None, defaultdict(int)
    for fname in files:
        p = hf_hub_download(IMAGENET_REPO, fname, repo_type="dataset")
        pf = pq.ParquetFile(p)
        if names is None:
            feats = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]
            names = imagenet_names(feats["label"]["names"][:1000])
        labels = pq.read_table(p, columns=["label"]).column("label").to_pylist()
        keep = []
        for i, l in enumerate(labels):
            if l < 1000 and (split != "train" or count[l] < per_class):
                count[l] += 1
                keep.append(i)
        if keep:
            table = pq.read_table(p, columns=["image"]).take(keep)
            xs.append(_decode(table.column("image").to_pylist()))
            ys.append(torch.tensor([labels[i] for i in keep]))
        print(f"  {fname}: kept {len(keep)} images", flush=True)
    out = {"x": torch.cat(xs), "y": torch.cat(ys), "names": names}
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(out, path)
    return out


class Cifar100:
    def __init__(self, cache_dir="data", device="cpu", features=False):
        """features=True swaps the images for precomputed CLIP image features (see t2v.features)."""
        tr, te = _load_split("train", cache_dir), _load_split("test", cache_dir)
        self.names = tr["names"]
        if features:
            tr["x"] = torch.load(os.path.join(cache_dir, "clip_feats_train.pt")).float()
            te["x"] = torch.load(os.path.join(cache_dir, "clip_feats_test.pt")).float()
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
        lut = torch.full((len(self.names),), -1, dtype=torch.long, device=y.device)
        lut[cls] = torch.arange(len(classes), device=y.device)
        return x[mask], lut[y[mask]]


class ImageNet32(Cifar100):
    def __init__(self, cache_dir="data", device="cpu", per_class=260):
        tr, te = _load_imagenet("train", cache_dir, per_class), _load_imagenet("test", cache_dir, per_class)
        self.names = tr["names"]
        self.train_x, self.train_y = tr["x"].to(device), tr["y"].to(device)
        self.test_x, self.test_y = te["x"].to(device), te["y"].to(device)
        self.coarse = None  # semantic clusters are computed from CLIP text embeddings (see cluster_classes)


def load_dataset(name="cifar100", cache_dir="data", device="cpu", features=False, per_class=260):
    if name == "cifar100":
        return Cifar100(cache_dir, device, features)
    if name == "imagenet32":
        return ImageNet32(cache_dir, device, per_class)
    raise ValueError(name)


def dataset_from_meta(meta, cache_dir, device, features=False):
    return load_dataset(meta.get("dataset", "cifar100"), cache_dir, device, features, meta.get("per_class", 260))


def prep(x, train=False):
    """Model input: uint8 images are normalised (and augmented when training); features pass through."""
    if x.dtype != torch.uint8:
        return x
    x = normalize(x)
    return augment(x) if train else x


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


def cluster_classes(emb, k, seed=0, iters=50):
    """k-means (cosine) over class text embeddings -> one cluster id per class. Used as 'superclasses'
    for task sampling and for the hard holdout, where whole semantic groups are held out."""
    g = torch.Generator().manual_seed(seed)
    emb = F.normalize(emb.float().cpu(), dim=-1)
    cent = emb[torch.randperm(len(emb), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = (emb @ cent.T).argmax(1)
        for c in range(k):
            m = assign == c
            cent[c] = F.normalize(emb[m].mean(0), dim=0) if m.any() else emb[torch.randint(len(emb), (1,), generator=g)][0]
    return (emb @ cent.T).argmax(1).tolist()


def split_classes(num_holdout, seed=0, num_classes=100, hard=0, clusters=None):
    """Seen classes + held-out classes. `hard` of the held-out classes are whole semantic clusters (so no
    close relative stays in the seen set); the remaining `num_holdout - hard` are drawn at random.
    Returns seen, holdout_random, holdout_hard (sorted lists)."""
    rng = random.Random(seed)
    hard_set = []
    if hard:
        by = defaultdict(list)
        for c, cl in enumerate(clusters):
            by[cl].append(c)
        order = list(by)
        rng.shuffle(order)
        for cl in order:
            if len(hard_set) >= hard:
                break
            if len(hard_set) + len(by[cl]) <= hard + 10:  # whole clusters only, allowing a small overshoot
                hard_set += by[cl]
    rest = [c for c in range(num_classes) if c not in set(hard_set)]
    rng.shuffle(rest)
    n_rand = num_holdout - len(hard_set) if hard else num_holdout
    rand_set = rest[:max(0, n_rand)]
    seen = rest[max(0, n_rand):]
    return sorted(seen), sorted(rand_set), sorted(hard_set)


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
