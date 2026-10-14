# T2V · إعادة تدريب المولّد بالتحسينات (خلية واحدة، تستخدم حديقة النماذج الموجودة في Drive)
# T2V-Hybrid: RPG prototypes + Tina-style joint transformer over all noisy tokens + per-token MLP denoiser
OUT = "/content/drive/MyDrive/t2v_resnet20_fast"
GEN_STEPS = 3000
VERSION = "v5_hybrid"

import json, os, subprocess, sys, time


def sh(cmd):
    print(f"\n>> {cmd}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "Warning" not in line and "Loading weights" not in line and "TransformerEncoder" not in line:
            print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"failed after {time.time() - t0:.0f}s: {cmd}")
    print(f"<< done in {time.time() - t0:.0f}s", flush=True)


FILES = {
    "t2v/__init__.py": r'''''',
    "t2v/data.py": r'''"""CIFAR-100 (loaded from the Hugging Face parquet mirror) and Tina-style task sampling."""
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
        lut = torch.full((100,), -1, dtype=torch.long, device=y.device)
        lut[cls] = torch.arange(len(classes), device=y.device)
        return x[mask], lut[y[mask]]


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
''',
    "t2v/target.py": r'''"""Vision models whose weights get generated. Every model ends in a Linear named `fc`."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_CLASSES = 10
HEAD = ("fc.weight", "fc.bias")


class TinaCNN(nn.Module):
    """~5.4K parameters, the size of Tina's CNN; every parameter is generated."""

    def __init__(self, num_classes=MAX_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.mean((2, 3)))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, widths, blocks, num_classes=MAX_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(F.relu(self.bn1(self.conv1(x))))
        return self.fc(x.mean((2, 3)))


class ClipHead(nn.Module):
    """Classifier head over frozen CLIP ViT-B/32 image features (Tina's ViT setting)."""

    def __init__(self, num_classes=MAX_CLASSES, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, feats):
        return self.fc(feats)


ARCHS = {
    "clip_head": lambda k: ClipHead(k),                                   # ~5.1K params on frozen CLIP
    "clip_zs": lambda k: ClipHead(k),                                     # same, generated as a residual on CLIP zero-shot
    "tina_cnn": lambda k: TinaCNN(k),                                     # ~5.4K params
    "resnet20": lambda k: ResNet((16, 32, 64), (3, 3, 3), k),             # ~0.27M params
    "resnet18": lambda k: ResNet((64, 128, 256, 512), (2, 2, 2, 2), k),   # ~11M params
}


def uses_features(arch):
    return arch in ("clip_head", "clip_zs")


ZS_SCALE = 50.0  # zero-shot head = ZS_SCALE * text embedding (logits = scaled cosine similarity)


def has_prior(arch):
    return arch == "clip_zs"


def add_head_prior(arch, flat, text_emb, sign=1.0):
    """For `clip_zs`, the generator works on W - ZS_SCALE * T: a correction to CLIP's zero-shot head.
    Adds (sign=+1) or removes (sign=-1) the zero-shot rows for the k classes in `text_emb` (k, 512).
    An unseen class whose correction comes out near zero falls back to plain CLIP zero-shot."""
    if not has_prior(arch):
        return flat
    spec = param_spec(build(arch))
    o, offs = 0, {}
    for name, shape in spec:
        offs[name] = (o, shape)
        o += int(torch.Size(shape).numel())
    w0, (K, feat) = offs[HEAD[0]]
    k = text_emb.shape[0]
    flat = flat.clone()
    rows = flat[..., w0:w0 + K * feat].view(*flat.shape[:-1], K, feat)
    rows[..., :k, :] += sign * ZS_SCALE * text_emb.to(flat.device, flat.dtype)
    flat[..., w0:w0 + K * feat] = rows.reshape(*flat.shape[:-1], K * feat)
    return flat


def build(arch, num_classes=MAX_CLASSES):
    return ARCHS[arch](num_classes)


def param_spec(model):
    """(name, shape) of every floating-point tensor in the state dict (weights + BN statistics)."""
    return [(k, tuple(v.shape)) for k, v in model.state_dict().items() if v.dtype.is_floating_point]


def flatten(model):
    sd = model.state_dict()
    return torch.cat([sd[k].detach().float().reshape(-1).cpu() for k, _ in param_spec(model)])


def load_flat(model, flat):
    """Load a flat parameter vector (as produced by `flatten` or the generator) into `model`."""
    sd, i = {}, 0
    for name, shape in param_spec(model):
        n = int(torch.Size(shape).numel())
        t = flat[i:i + n].view(shape)
        if name.endswith("running_var"):
            t = t.clamp_min(1e-5)
        sd[name] = t
        i += n
    model.load_state_dict(sd, strict=False)
    return model
''',
    "t2v/text_embed.py": r'''"""Frozen CLIP text encoder: one embedding per class name (Tina's conditioning)."""
import torch
import torch.nn.functional as F

CLIP_ID = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo of a {}."
# CLIP's prompt-ensembling trick: averaging several templates gives steadier class embeddings.
TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a close-up photo of a {}.", "a low resolution photo of a {}.",
             "a bright photo of a {}.", "a cropped photo of a {}.", "a photo of the small {}.", "a photo of the large {}."]
_clip = {}


def _load(device):
    if device not in _clip:
        from transformers import CLIPModel, CLIPTokenizer
        model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        _clip[device] = (model, CLIPTokenizer.from_pretrained(CLIP_ID))
    return _clip[device]


@torch.no_grad()
def embed_names(names, device="cpu", ensemble=False):
    model, tok = _load(device)
    templates = TEMPLATES if ensemble else [TEMPLATE]
    embs = []
    for template in templates:
        texts = [template.format(n.replace("_", " ")) for n in names]
        batch = tok(texts, padding=True, return_tensors="pt").to(device)
        out = model.get_text_features(**batch)
        if not torch.is_tensor(out):  # newer transformers return a model output object
            out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
        embs.append(F.normalize(out.float(), dim=-1))
    return F.normalize(torch.stack(embs).mean(0), dim=-1)
''',
    "t2v/tokenizer.py": r'''"""Flat parameter vector <-> sequence of fixed-size tokens (RPG's layer-wise tokenisation).

Body tensors are normalised per tensor (mean/std over the whole zoo) and cut into `token_size`
chunks. The classifier is tokenised differently: one token per class row ([W_i, b_i]), so each
row can be tied to the text embedding of its class. That makes class permutation (Tina's key
augmentation) a permutation of tokens.
"""
import math

import torch

from .target import HEAD


class ParamTokenizer:
    def __init__(self, spec, token_size, stats, max_classes):
        self.spec, self.token_size, self.stats, self.K = spec, token_size, stats, max_classes
        offsets, o = {}, 0
        for name, shape in spec:
            offsets[name] = (o, shape)
            o += int(torch.Size(shape).numel())
        self.num_params = o

        idx, mean, std, layer_ids, intra_ids, slots = [], [], [], [], [], []

        elem_mean = stats.get("__elem_mean__")  # per-element zoo mean for body weights (norm="element"/"zscore")
        elem_std = stats.get("__elem_std__")    # per-element zoo spread (norm="zscore")
        if elem_mean is not None:
            elem_mean = elem_mean.cpu()  # checkpoints may be loaded straight onto the GPU
        if elem_std is not None:
            elem_std = elem_std.cpu()

        def add(positions, name, layer, intra, slot):
            pad = token_size - len(positions)
            idx.append(torch.cat([positions, torch.full((pad,), -1, dtype=torch.long)]))
            if elem_mean is not None and slot < 0:
                mean.append(torch.cat([elem_mean[positions].float(), torch.zeros(pad)]))
            else:
                mean.append(torch.full((token_size,), stats[name][0]))
            if elem_std is not None and slot < 0:
                std.append(torch.cat([elem_std[positions].float(), torch.ones(pad)]))
            else:
                std.append(torch.full((token_size,), stats[name][1]))
            layer_ids.append(layer), intra_ids.append(intra), slots.append(slot)

        body = [n for n, _ in spec if n not in HEAD]
        for li, name in enumerate(body):
            start, shape = offsets[name]
            n = int(torch.Size(shape).numel())
            for j in range(math.ceil(n / token_size)):
                lo, hi = j * token_size, min(n, (j + 1) * token_size)
                add(torch.arange(start + lo, start + hi), name, li, j, -1)

        (w0, (k, feat)), (b0, _) = offsets[HEAD[0]], offsets[HEAD[1]]
        assert k == max_classes and feat + 1 <= token_size, "classifier row must fit in one token"
        head_pos = []
        for i in range(k):
            head_pos.append(len(slots))
            row = torch.cat([torch.arange(w0 + i * feat, w0 + (i + 1) * feat), torch.tensor([b0 + i])])
            add(row, HEAD[0], len(body), 0, i)
            # the bias entry uses the bias statistics
            mean[-1][feat], std[-1][feat] = stats[HEAD[1]]

        self.idx = torch.stack(idx)                 # (T, D) index into the flat vector, -1 = padding
        self.valid = self.idx >= 0                  # (T, D)
        self.mean, self.std = torch.stack(mean), torch.stack(std)
        self.layer_ids = torch.tensor(layer_ids)
        self.intra_ids = torch.tensor(intra_ids)
        self.slots = torch.tensor(slots)            # class slot for head rows, -1 for body tokens
        self.head_pos = torch.tensor(head_pos)      # token positions of the K head rows
        self.num_tokens = len(slots)

    @staticmethod
    def compute_stats(flat, spec, norm="tensor", eps=1e-6):
        """norm="tensor": one mean/std per tensor (RPG). norm="element": body weights are expressed as a
        deviation from the zoo's mean model (scaled per tensor), so the generator spends its capacity on
        what differs between tasks. norm="zscore": the same deviation, scaled by each weight's own spread
        across the zoo, so a generation error of a given size is equally harmless for every weight (deep
        BatchNorm nets such as ResNet collapse once errors exceed ~2x the zoo spread). The classifier keeps
        per-tensor stats because its rows follow the classes, not fixed positions."""
        stats, o = {}, 0
        mean_model = flat.float().mean(0)
        if norm in ("element", "zscore"):
            stats["__elem_mean__"] = mean_model
        if norm == "zscore":
            stats["__elem_std__"] = (flat.float() - mean_model).std(0).clamp_min(1e-8)
        for name, shape in spec:
            n = int(torch.Size(shape).numel())
            t = flat[:, o:o + n].float()
            if norm in ("element", "zscore") and name not in HEAD:
                stats[name] = (0.0, max((t - mean_model[o:o + n]).std().item(), eps))
            else:
                stats[name] = (t.mean().item(), max(t.std().item(), eps))
            o += n
        return stats

    def to(self, device):
        for k in ("idx", "valid", "mean", "std", "layer_ids", "intra_ids", "slots", "head_pos"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def encode(self, flat):
        """(B, P) -> (B, T, D) normalised tokens, zero in padding."""
        x = flat.float()[:, self.idx.clamp_min(0)]
        return torch.where(self.valid, (x - self.mean) / self.std, torch.zeros((), device=x.device))

    def decode(self, tokens, num_classes=None):
        """(B, T, D) -> (B, P). Head rows beyond `num_classes` are zeroed."""
        x = tokens.float() * self.std + self.mean
        if num_classes is not None:
            unused = (self.slots >= num_classes)[None, :, None]
            x = torch.where(unused, torch.zeros((), device=x.device), x)
        flat = torch.zeros(tokens.shape[0], self.num_params, device=tokens.device)
        flat[:, self.idx[self.valid]] = x[:, self.valid]
        return flat

    def state(self):
        return {"spec": self.spec, "token_size": self.token_size, "stats": self.stats, "max_classes": self.K}

    @classmethod
    def from_state(cls, s):
        return cls(s["spec"], s["token_size"], s["stats"], s["max_classes"])
''',
    "t2v/generator.py": r'''"""Text-conditioned recurrent parameter diffusion: Tina's conditioning on RPG's scalable generator.

  class names --CLIP--> class tokens --set encoder--> condition
  token positions (layer, chunk) + condition + per-row class embedding
      --recurrent backbone (LSTM / GRU / Transformer / Mamba)--> one prototype per parameter token
  every token is then denoised independently by a small 1-D conv U-Net conditioned on its prototype,
  so memory grows with the number of tokens processed per step, not with the model size.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos(pos, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=pos.device) / half)
    ang = pos.float()[:, None] * freq[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class CondEncoder(nn.Module):
    """Set encoder over class embeddings. No positional encoding, so it is permutation invariant;
    missing classes (tasks with fewer than K classes) become a learned placeholder, as in Tina."""

    def __init__(self, clip_dim, d, layers, heads):
        super().__init__()
        self.inp = nn.Linear(clip_dim, d)
        self.placeholder = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, emb, cls_mask):
        h = torch.where(cls_mask[..., None], self.inp(emb), self.placeholder)
        return self.enc(h)


class Backbone(nn.Module):
    def __init__(self, kind, d, layers, heads):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.net = getattr(nn, kind.upper())(d, d, layers, batch_first=True)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.net = nn.TransformerEncoder(layer, layers)
        elif kind == "mamba":  # RPG's default; needs `pip install mamba-ssm` (has ROCm support)
            from mamba_ssm import Mamba
            self.net = nn.ModuleList([Mamba(d_model=d) for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        else:
            raise ValueError(kind)

    def forward(self, h):
        if self.kind in ("lstm", "gru"):
            return h + self.net(h)[0]
        if self.kind == "transformer":
            return self.net(h)
        for norm, layer in zip(self.norms, self.net):
            h = h + layer(norm(h))
        return h


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout, d):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 3, padding=1)
        self.film = nn.Linear(d, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(cond)[..., None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return h + self.skip(x)


class ConvDenoiser(nn.Module):
    """1-D U-Net over one parameter token, conditioned on (prototype + timestep) via FiLM."""

    def __init__(self, token_size, d, channels):
        super().__init__()
        assert token_size % 2 ** (len(channels) - 1) == 0
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.d = d
        self.inp = nn.Conv1d(1, channels[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        cin = channels[0]
        for i, c in enumerate(channels):
            self.down.append(FiLMBlock(cin, c, d))
            self.pool.append(nn.Conv1d(c, c, 3, stride=2, padding=1) if i < len(channels) - 1 else nn.Identity())
            cin = c
        self.mid = FiLMBlock(cin, cin, d)
        self.up = nn.ModuleList()
        for c in reversed(channels):
            self.up.append(FiLMBlock(cin + c, c, d))
            cin = c
        self.out = nn.Conv1d(cin, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, proto):
        cond = proto + self.time(sincos(t, self.d))
        h, skips = self.inp(x[:, None]), []
        for block, pool in zip(self.down, self.pool):
            h = block(h, cond)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, cond)
        for block in self.up:
            s = skips.pop()
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = block(torch.cat([h, s], 1), cond)
        return self.out(h)[:, 0]


class MLPDenoiser(nn.Module):
    """Residual MLP over one parameter token with FiLM conditioning; much cheaper than the U-Net
    for small tokens (and on CPU)."""

    def __init__(self, token_size, d, hidden=512, depth=3):
        super().__init__()
        self.d = d
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.inp = nn.Linear(token_size, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden, elementwise_affine=False) for _ in range(depth)])
        self.films = nn.ModuleList([nn.Linear(d, 2 * hidden) for _ in range(depth)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
                                   for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, token_size))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, proto):
        cond = F.silu(proto + self.time(sincos(t, self.d)))
        h = self.inp(x)
        for norm, film, mlp in zip(self.norms, self.films, self.mlps):
            scale, shift = film(cond).chunk(2, -1)
            h = h + mlp(norm(h) * (1 + scale) + shift)
        return self.out(h)


def cosine_alpha_bar(steps, s=0.008):
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    f = torch.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(0, 0.999)
    return torch.cumprod(1 - betas, 0).float()


class T2VGenerator(nn.Module):
    def __init__(self, layer_ids, intra_ids, slots, token_size, clip_dim=512, d_model=512, backbone="lstm",
                 backbone_layers=2, cond_layers=2, heads=8, channels=(64, 128, 256), diffusion_steps=1000,
                 denoiser="conv", mlp_hidden=512, mlp_depth=3, pred="x0", clip=None, joint_layers=0,
                 valid=None):
        super().__init__()
        self.config = dict(token_size=token_size, clip_dim=clip_dim, d_model=d_model, backbone=backbone,
                           backbone_layers=backbone_layers, cond_layers=cond_layers, heads=heads,
                           channels=tuple(channels), diffusion_steps=diffusion_steps, denoiser=denoiser,
                           mlp_hidden=mlp_hidden, mlp_depth=mlp_depth, pred=pred, clip=clip,
                           joint_layers=joint_layers)
        self.register_buffer("slots", slots.clone(), persistent=False)
        half = d_model // 2
        self.register_buffer("pos", torch.cat([sincos(layer_ids, half), sincos(intra_ids, half)], -1), persistent=False)
        self.register_buffer("alpha_bar", cosine_alpha_bar(diffusion_steps), persistent=False)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.cond = CondEncoder(clip_dim, d_model, cond_layers, heads)
        self.backbone = Backbone(backbone, d_model, backbone_layers, heads)
        self.xattn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.xnorm = nn.LayerNorm(d_model)
        self.proto_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(),
                                       nn.Linear(d_model, d_model))
        self.denoiser = (ConvDenoiser(token_size, d_model, channels) if denoiser == "conv"
                         else MLPDenoiser(token_size, d_model, mlp_hidden, mlp_depth))
        # Hybrid mode (joint_layers > 0): as in Tina / G.pt, every noisy parameter token of a model is
        # embedded and a transformer lets the tokens attend to each other (and to the class tokens) at each
        # denoising step, so the layers of one network are denoised consistently. Its output refines the
        # RPG prototypes, and the cheap per-token denoiser above still decodes each token.
        if joint_layers > 0:
            if valid is None:
                raise ValueError("joint mode needs the tokenizer's valid mask")
            self.register_buffer("valid", valid.clone(), persistent=False)
            self.joint_in = nn.Linear(token_size, d_model)
            self.joint_time = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, dropout=0.0, batch_first=True,
                                               norm_first=True)
            self.joint = nn.TransformerDecoder(layer, joint_layers)  # self-attention over tokens + cross-attention to classes
            self.joint_out = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.joint_out.weight)
            nn.init.zeros_(self.joint_out.bias)

    def prototypes(self, emb, cls_mask, return_classes=False):
        c = self.cond(emb, cls_mask)                                             # (B, K, d)
        m = cls_mask[..., None].float()
        pooled = (c * m).sum(1) / m.sum(1).clamp_min(1)                         # (B, d)
        h = self.pos_proj(self.pos)[None] + pooled[:, None]                      # (B, T, d)
        is_head = (self.slots >= 0)[None, :, None]
        h = h + torch.where(is_head, c[:, self.slots.clamp_min(0)], torch.zeros((), device=h.device))
        h = self.backbone(h)
        h = h + self.xattn(self.xnorm(h), c, c, need_weights=False)[0]
        return (self.proto_out(h), c) if return_classes else self.proto_out(h)

    def context(self, x_t, step, proto, classes):
        """Joint mode: refine prototypes with the current noisy tokens of the whole network.
        x_t (B,T,D), step (B,), proto (B,T,d), classes (B,K,d) -> (B,T,d)."""
        h = self.joint_in(x_t * self.valid) + proto + self.joint_time(sincos(step, self.config["d_model"]))[:, None]
        return proto + self.joint_out(self.joint(h, classes))

    def loss(self, x0, valid, emb, cls_mask, row_valid, tokens_per_step=None, min_snr=0.0, head_weight=1.0):
        """x0 (B,T,D) normalised tokens, valid (T,D), row_valid (B,T) -> denoising MSE.
        min_snr > 0 applies Min-SNR-gamma weighting (Hang et al., 2023), which down-weights the very
        noisy timesteps where the target is unpredictable and speeds up convergence.
        Classifier-row tokens are always included and weighted by head_weight: in deep nets they are a
        tiny fraction of all tokens (10 of 364 in ResNet-20), yet they carry all the class information."""
        joint = self.config.get("joint_layers", 0) > 0
        proto, classes = self.prototypes(emb, cls_mask, return_classes=True)
        if joint:
            B = x0.shape[0]
            step_b = torch.randint(0, len(self.alpha_bar), (B,), device=x0.device)
            ab_b = self.alpha_bar[step_b][:, None, None]
            noise_all = torch.randn_like(x0)
            xt_all = ab_b.sqrt() * x0 + (1 - ab_b).sqrt() * noise_all
            proto = self.context(xt_all, step_b, proto, classes)
        sel = row_valid.nonzero()
        is_head = self.slots[sel[:, 1]] >= 0
        if tokens_per_step and len(sel) > tokens_per_step:
            head, body = sel[is_head], sel[~is_head]
            keep = max(0, tokens_per_step - len(head))
            body = body[torch.randperm(len(body), device=sel.device)[:keep]]
            sel = torch.cat([head, body])
            is_head = self.slots[sel[:, 1]] >= 0
        b, t = sel[:, 0], sel[:, 1]
        x, p, vm = x0[b, t], proto[b, t], valid[t].float()
        if joint:
            step, noise, xt = step_b[b], noise_all[b, t], xt_all[b, t]
            ab = self.alpha_bar[step][:, None]
        else:
            step = torch.randint(0, len(self.alpha_bar), (len(x),), device=x.device)
            ab = self.alpha_bar[step][:, None]
            noise = torch.randn_like(x)
            xt = ab.sqrt() * x + (1 - ab).sqrt() * noise
        pred = self.denoiser(xt, step, p)
        target = x if self.config["pred"] == "x0" else noise
        err = (((pred.float() - target) ** 2) * vm).sum(-1)
        if min_snr > 0:
            snr = (ab / (1 - ab)).squeeze(-1)
            w = snr.clamp(max=min_snr) / min_snr          # x0-prediction form
            if self.config["pred"] != "x0":
                w = w * min_snr / snr.clamp(min=1e-8)     # eps-prediction form: min(snr, g) / snr
            err = err * w
        if head_weight != 1.0:
            hw = torch.where(is_head, torch.full_like(err, head_weight), torch.ones_like(err))
            return (err * hw).sum() / (vm.sum(-1) * hw).sum()
        return err.sum() / vm.sum()

    @torch.no_grad()
    def sample(self, emb, cls_mask, steps=100, chunk=4096, clip="auto"):
        """DDIM (eta=0). Returns (B, T, D) normalised tokens. `clip` bounds the x0 estimate; "auto" uses
        the largest normalised value seen in the zoo (without it, eps-prediction diverges at t~T)."""
        if clip == "auto":
            clip = self.config.get("clip")
        joint = self.config.get("joint_layers", 0) > 0
        proto, classes = self.prototypes(emb, cls_mask, return_classes=True)
        B, T, d = proto.shape
        p = proto.reshape(B * T, d)
        x = torch.randn(B * T, self.config["token_size"], device=p.device)
        ts = torch.linspace(len(self.alpha_bar) - 1, 0, steps, device=p.device).round().long()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=p.device)
            if joint:  # the whole network's current noisy state refines every token's condition
                p = self.context(x.view(B, T, -1), t.expand(B), proto, classes).reshape(B * T, d)
            out = torch.cat([self.denoiser(x[j:j + chunk], t.expand(len(x[j:j + chunk])), p[j:j + chunk]).float()
                             for j in range(0, len(x), chunk)])
            if self.config.get("pred", "eps") == "x0":
                x0 = out
            else:
                x0 = (x - (1 - ab).sqrt() * out) / ab.sqrt()
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.view(B, T, -1)
''',
    "t2v/zoo.py": r'''"""Build the model zoo the generator learns from (Tina's recipe).

Stage 1  `generic`: one model trained on every seen class (100-way head).
Stage 2  `pmodels`: many personalised 10-way models, each fine-tuned from the generic model on one
         task (a set of classes). Every p-model is stored as a flat parameter vector plus its class ids.

Shards let several processes share one MI300X (or several GPUs):
    for s in 0 1 2 3; do python -m t2v.zoo pmodels --shard $s --num_shards 4 & done
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from . import target
from .data import Cifar100, prep, sample_task, split_classes


def pick_device(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"  # ROCm PyTorch also reports "cuda"


def train(model, x_u8, y, epochs, lr, bs, k=None, wd=5e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    total = max(1, epochs * math.ceil(len(y) / bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=y.device)
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            logits = model(prep(x_u8[idx], train=True))
            if k is not None:
                logits = logits[:, :k]  # padded class slots never receive gradient
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, x_u8, y, k=None, bs=2000):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        logits = model(prep(x_u8[i:i + bs]))
        if k is not None:
            logits = logits[:, :k]
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    return correct / max(1, len(y))


def cmd_generic(a):
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    data = Cifar100(a.data, dev, features=target.uses_features(a.arch))
    seen, holdout = split_classes(a.num_holdout, a.seed)
    x, y = data.subset(seen)
    y = torch.tensor(seen, device=dev)[y]  # keep original 0..99 labels for the 100-way head
    model = target.build(a.arch, 100).to(dev)
    t0 = time.time()
    train(model, x, y, a.epochs, a.lr, a.bs, wd=a.wd)
    xt, yt = data.subset(seen, train=False)
    acc = accuracy(model, xt, torch.tensor(seen, device=dev)[yt])
    os.makedirs(a.out, exist_ok=True)
    torch.save({"arch": a.arch, "state": model.state_dict()}, os.path.join(a.out, "generic.pt"))
    meta = {"arch": a.arch, "seen": seen, "holdout": holdout, "names": data.names, "coarse": data.coarse,
            "max_classes": target.MAX_CLASSES, "num_params": sum(int(torch.Size(s).numel()) for _, s in
                                                                target.param_spec(target.build(a.arch)))}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"generic {a.arch}: {acc:.3f} test acc on {len(seen)} seen classes, {time.time() - t0:.0f}s")


def personalise(generic_state, arch, classes, dev, text_emb=None):
    """10-way model initialised from the generic one; head rows copied for the task's classes
    (for `clip_zs`: initialised from CLIP's zero-shot head instead)."""
    m = target.build(arch).to(dev)
    body = {k: v for k, v in generic_state.items() if not k.startswith("fc.")}
    m.load_state_dict(body, strict=False)
    with torch.no_grad():
        m.fc.weight.zero_()
        m.fc.bias.zero_()
        if target.has_prior(arch):
            m.fc.weight[:len(classes)] = target.ZS_SCALE * text_emb[classes]
        else:
            m.fc.weight[:len(classes)] = generic_state["fc.weight"][classes]
            m.fc.bias[:len(classes)] = generic_state["fc.bias"][classes]
    return m


def cmd_pmodels(a):
    dev = pick_device(a.device)
    with open(os.path.join(a.out, "meta.json")) as f:
        meta = json.load(f)
    g = torch.load(os.path.join(a.out, "generic.pt"), map_location=dev)
    arch, K = g["arch"], meta["max_classes"]
    data = Cifar100(a.data, dev, features=target.uses_features(arch))
    seen, holdout = meta["seen"], meta["holdout"]
    text_emb = None
    if target.has_prior(arch):
        from .text_embed import embed_names
        text_emb = embed_names(meta["names"], dev, ensemble=True)

    # Tasks are a pure function of (split, seed, index), so shards never overlap and reruns resume.
    rng = random.Random(f"{a.split}-{a.seed}")
    tasks = []
    for _ in range(a.n):
        if a.split in ("eval_unseen", "eval_novel"):  # eval_novel: every class held out (Tina's 100% unseen)
            n_new = K if a.split == "eval_novel" else max(1, round(a.unseen_frac * K))
            t = rng.sample(holdout, n_new) + rng.sample(seen, K - n_new)
            rng.shuffle(t)
        else:
            t = sample_task(rng, seen, meta["coarse"], a.min_k, K)
        tasks.append(t)
    if a.split == "eval_seen":  # unseen combinations of seen classes
        train_rng = random.Random(f"train-{a.seed}")
        train_sets = {frozenset(sample_task(train_rng, seen, meta["coarse"], a.min_k, K)) for _ in range(a.n_train_ref)}
        tasks = [t for t in tasks if frozenset(t) not in train_sets]

    path = os.path.join(a.out, f"{a.split}_shard{a.shard}.pt")
    done = torch.load(path) if os.path.exists(path) else {"flat": [], "classes": [], "acc": [], "index": []}
    have = set(done["index"])
    mine = [i for i in range(len(tasks)) if i % a.num_shards == a.shard and i not in have]
    print(f"{a.split} shard {a.shard}: {len(mine)} p-models to train ({len(have)} already done)")
    t0 = time.time()
    for n, i in enumerate(mine):
        torch.manual_seed(a.seed * 1_000_003 + ["train", "eval_seen", "eval_unseen", "eval_novel"].index(a.split) * 100_003 + i)
        classes = tasks[i]
        m = personalise(g["state"], arch, classes, dev, text_emb)
        if a.finetune == "head":
            for name, p in m.named_parameters():
                p.requires_grad_(name.startswith("fc."))
        x, y = data.subset(classes)
        train(m, x, y, a.epochs, a.lr, a.bs, k=len(classes), wd=a.wd)
        xt, yt = data.subset(classes, train=False)
        flat = target.flatten(m)
        if text_emb is not None:  # store the correction to the zero-shot head, not the raw head
            flat = target.add_head_prior(arch, flat, text_emb[classes].cpu(), sign=-1.0)
        done["flat"].append(flat.half() if a.fp16 else flat)
        done["classes"].append(classes + [-1] * (K - len(classes)))
        done["acc"].append(accuracy(m, xt, yt, k=len(classes)))
        done["index"].append(i)
        if (n + 1) % a.save_every == 0 or n + 1 == len(mine):
            torch.save(done, path)
            rate = (n + 1) / (time.time() - t0)
            print(f"  {n + 1}/{len(mine)}  mean acc {sum(done['acc']) / len(done['acc']):.3f}  {rate:.2f} models/s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generic", "pmodels"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="zoo")
        s.add_argument("--data", default="data")
        s.add_argument("--device", default="auto")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--bs", type=int, default=128)
    g = sub.choices["generic"]
    g.add_argument("--arch", default="resnet20", choices=list(target.ARCHS))
    g.add_argument("--num_holdout", type=int, default=20, help="classes never used by the zoo")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--wd", type=float, default=5e-4)
    m = sub.choices["pmodels"]
    m.add_argument("--split", default="train", choices=["train", "eval_seen", "eval_unseen", "eval_novel"])
    m.add_argument("--n", type=int, default=1000, help="number of tasks for this split")
    m.add_argument("--n_train_ref", type=int, default=1000, help="train-split size, to exclude its tasks from eval_seen")
    m.add_argument("--min_k", type=int, default=10, help="smallest class count per task (Tina pads to 10)")
    m.add_argument("--unseen_frac", type=float, default=0.5, help="eval_unseen: fraction of held-out classes per task")
    m.add_argument("--epochs", type=int, default=10)
    m.add_argument("--lr", type=float, default=0.01)
    m.add_argument("--wd", type=float, default=5e-4)
    m.add_argument("--finetune", default="all", choices=["all", "head"])
    m.add_argument("--fp16", action="store_true", help="store p-models in fp16 (halves zoo size)")
    m.add_argument("--shard", type=int, default=0)
    m.add_argument("--num_shards", type=int, default=1)
    m.add_argument("--save_every", type=int, default=25)
    a = p.parse_args()
    {"generic": cmd_generic, "pmodels": cmd_pmodels}[a.cmd](a)


if __name__ == "__main__":
    main()
''',
    "t2v/train_gen.py": r'''"""Train the text-to-weights generator on a model zoo built by `t2v.zoo`."""
import argparse
import glob
import json
import math
import os
import signal
import time

import torch

from .generator import T2VGenerator
from .text_embed import embed_names
from .tokenizer import ParamTokenizer
from .zoo import pick_device
from . import target


def load_split(zoo, split):
    flats, classes, accs = [], [], []
    for path in sorted(glob.glob(os.path.join(zoo, f"{split}_shard*.pt"))):
        s = torch.load(path)
        flats += s["flat"]
        classes += s["classes"]
        accs += s["acc"]
    if not flats:
        raise SystemExit(f"no {split} shards in {zoo}")
    return torch.stack(flats).float(), torch.tensor(classes), torch.tensor(accs)


def class_batch(class_emb, classes):
    """classes (B,K) with -1 padding -> (emb (B,K,E), mask (B,K), k (B,))."""
    mask = classes >= 0
    return class_emb[classes.clamp_min(0)], mask, mask.sum(1)


def permute_classes(x0, classes, k, head_pos):
    """Tina's classifier augmentation: shuffle class order and the matching head rows together."""
    B, K = classes.shape
    keys = torch.rand(B, K, device=x0.device)
    keys[torch.arange(K, device=x0.device)[None] >= k[:, None]] = 2.0  # padding stays at the end
    perm = keys.argsort(1)
    rows = x0[torch.arange(B, device=x0.device)[:, None], head_pos[perm]]
    x0 = x0.clone()
    x0[:, head_pos] = rows
    return x0, classes.gather(1, perm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zoo", default="zoo")
    p.add_argument("--out", default="ckpt/generator.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--token_size", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--backbone", default="lstm", choices=["lstm", "gru", "transformer", "mamba"])
    p.add_argument("--backbone_layers", type=int, default=2)
    p.add_argument("--cond_layers", type=int, default=2)
    p.add_argument("--channels", default="64,128,256", help="conv denoiser widths")
    p.add_argument("--denoiser", default="conv", choices=["conv", "mlp"])
    p.add_argument("--mlp_hidden", type=int, default=512)
    p.add_argument("--mlp_depth", type=int, default=3)
    p.add_argument("--norm", default="tensor", choices=["tensor", "element", "zscore"],
                   help="body weight normalisation (zscore recommended for BatchNorm nets like ResNet)")
    p.add_argument("--pred", default="x0", choices=["x0", "eps"], help="denoiser target (x0 = Tina/G.pt signal prediction)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--tokens_per_step", type=int, default=8192, help="tokens denoised per step (memory knob)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--cosine", action="store_true", help="cosine-decay the LR to 0 after warmup")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay for the saved weights (e.g. 0.999); 0 = off")
    p.add_argument("--no_perm_aug", action="store_true")
    p.add_argument("--prompt_ensemble", action="store_true", help="average CLIP embeddings over several templates")
    p.add_argument("--emb_noise", type=float, default=0.0, help="std of Gaussian noise added to class embeddings")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (recommended on MI300X)")
    p.add_argument("--fp16", action="store_true", help="fp16 autocast + loss scaling (for T4/V100-class GPUs)")
    p.add_argument("--joint_layers", type=int, default=0,
                   help="hybrid mode: Tina/G.pt-style transformer layers over all noisy tokens (0 = pure RPG)")
    p.add_argument("--head_weight", type=float, default=1.0,
                   help="loss weight of classifier-row tokens (they are also always included in each step)")
    p.add_argument("--min_snr", type=float, default=0.0, help="Min-SNR-gamma loss weighting, e.g. 5; 0 = off")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000,
                   help="checkpoint period; also: kill -USR1 <pid> saves now, Ctrl-C / kill saves and stops")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    with open(os.path.join(a.zoo, "meta.json")) as f:
        meta = json.load(f)
    flat, classes, accs = load_split(a.zoo, "train")
    print(f"zoo: {len(flat)} p-models x {flat.shape[1]:,} params (mean acc {accs.mean():.3f})")

    spec = target.param_spec(target.build(meta["arch"]))
    tok = ParamTokenizer(spec, a.token_size, ParamTokenizer.compute_stats(flat, spec, a.norm), meta["max_classes"]).to(dev)
    class_emb = embed_names(meta["names"], dev, ensemble=a.prompt_ensemble)
    clip = max(tok.encode(flat[i:i + 256].to(dev)).abs().max().item() for i in range(0, len(flat), 256))
    gen = T2VGenerator(tok.layer_ids, tok.intra_ids, tok.slots, a.token_size, class_emb.shape[1], a.d_model,
                       a.backbone, a.backbone_layers, a.cond_layers,
                       channels=[int(c) for c in a.channels.split(",")], denoiser=a.denoiser,
                       mlp_hidden=a.mlp_hidden, mlp_depth=a.mlp_depth, pred=a.pred, clip=round(clip, 3),
                       joint_layers=a.joint_layers, valid=tok.valid).to(dev)
    n_gen = sum(p.numel() for p in gen.parameters())
    print(f"tokens per model: {tok.num_tokens} x {a.token_size}; generator: {n_gen / 1e6:.1f}M params")

    opt = torch.optim.AdamW(gen.parameters(), lr=a.lr, weight_decay=0.0)
    def lr_lambda(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        if a.cosine:
            return 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = None
    if a.ema > 0:
        ema = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    flat = flat.pin_memory() if dev.startswith("cuda") else flat
    use_fp16 = a.fp16 and dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def save(step):
        weights = ema if ema is not None else gen.state_dict()
        torch.save({"generator": weights, "config": gen.config, "tokenizer": tok.state(),
                    "arch": meta["arch"], "names": meta["names"], "prompt_ensemble": a.prompt_ensemble,
                    "step": step}, a.out)

    flags = {"stop": False, "snapshot": False}
    signal.signal(signal.SIGINT, lambda *_: flags.update(stop=True))
    signal.signal(signal.SIGTERM, lambda *_: flags.update(stop=True))
    signal.signal(signal.SIGUSR1, lambda *_: flags.update(snapshot=True))

    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(flat), (a.bs,))
        x0 = tok.encode(flat[idx].to(dev, non_blocking=True))
        cls = classes[idx].to(dev)
        k = (cls >= 0).sum(1)
        if not a.no_perm_aug:
            x0, cls = permute_classes(x0, cls, k, tok.head_pos)
        emb, mask, k = class_batch(class_emb, cls)
        if a.emb_noise > 0:
            emb = torch.nn.functional.normalize(emb + a.emb_noise * torch.randn_like(emb), dim=-1)
        row_valid = (tok.slots[None] < 0) | (tok.slots[None] < k[:, None])
        with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=amp_dtype,
                            enabled=a.bf16 or use_fp16):
            loss = gen.loss(x0, tok.valid, emb, mask, row_valid, a.tokens_per_step, a.min_snr, a.head_weight)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].lerp_(v, 1 - a.ema)
                    else:
                        ema[k].copy_(v)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.log_every == 0:
            print(f"step {step}  loss {run:.4f}  {step / (time.time() - t0):.1f} it/s", flush=True)
        if step % a.save_every == 0 or flags["snapshot"]:
            save(step)
            flags["snapshot"] = False
        if flags["stop"]:
            save(step)
            print(f"stopped at step {step}, saved {a.out}")
            return
    save(a.steps)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
''',
    "t2v/generate.py": r'''"""Text in, vision model out: `python -m t2v.generate --prompt "cats, dogs and horses"`."""
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
''',
    "t2v/evaluate.py": r'''"""Accuracy of generated models on held-out tasks, against Tina's baselines.

  generated (mean / best of n)  weights straight from the generator, no training, no images
  oracle                        a p-model actually fine-tuned on the task (upper reference)
  selection                     the generic model's head restricted to the task's classes
                                (impossible when a class was never seen, like TAPER in Tina)
"""
import argparse
import json
import os
import time

import torch

from . import target
from .data import Cifar100, prep
from .generate import generate_flat, load_generator
from .text_embed import embed_names
from .train_gen import load_split
from .zoo import accuracy, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/generator.pt")
    p.add_argument("--zoo", default="zoo")
    p.add_argument("--data", default="data")
    p.add_argument("--splits", default="eval_seen,eval_unseen,eval_novel")
    p.add_argument("--max_tasks", type=int, default=100)
    p.add_argument("--samples", type=int, default=5, help="generated models per task")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/eval.json")
    a = p.parse_args()

    dev = pick_device(a.device)
    gen, tok, ck = load_generator(a.ckpt, dev)
    arch = ck["arch"]
    data = Cifar100(a.data, dev, features=target.uses_features(arch))
    generic = target.build(arch, 100).to(dev)
    generic.load_state_dict(torch.load(os.path.join(a.zoo, "generic.pt"), map_location=dev)["state"])
    with open(os.path.join(a.zoo, "meta.json")) as f:
        seen = set(json.load(f)["seen"])
    class_emb = embed_names(ck["names"], dev, ensemble=ck.get("prompt_ensemble", False))
    net = target.build(arch).to(dev)
    zs_emb = embed_names(ck["names"], dev, ensemble=True)  # CLIP zero-shot uses the prompt ensemble

    report = {}
    for split in a.splits.split(","):
        try:
            flat, classes, oracle = load_split(a.zoo, split)
        except SystemExit as e:
            print(e)
            continue
        rows, gen_time = [], 0.0
        for i in range(min(a.max_tasks, len(flat))):
            cls = [c for c in classes[i].tolist() if c >= 0]
            xt, yt = data.subset(cls, train=False)
            t0 = time.time()
            flats = generate_flat(gen, tok, cls, n=a.samples, steps=a.steps, emb=class_emb[cls],
                                  prior_emb=zs_emb[cls], arch=arch)
            gen_time += time.time() - t0
            accs = [accuracy(target.load_flat(net, f), xt, yt, k=len(cls)) for f in flats]
            sel = None
            if all(c in seen for c in cls):
                with torch.no_grad():
                    logits = generic(prep(xt))[:, cls]
                sel = (logits.argmax(1) == yt).float().mean().item()
            zs = None
            if target.uses_features(arch):
                zs = ((xt @ zs_emb[cls].T).argmax(1) == yt).float().mean().item()
            rows.append({"classes": cls, "generated": accs, "oracle": oracle[i].item(), "selection": sel,
                         "clip_zero_shot": zs})
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        summary = {
            "tasks": len(rows),
            "generated_mean": mean([mean(r["generated"]) for r in rows]),
            "generated_best": mean([max(r["generated"]) for r in rows]),
            "oracle": mean([r["oracle"] for r in rows]),
            "selection": mean([r["selection"] for r in rows if r["selection"] is not None]),
            "clip_zero_shot": mean([r["clip_zero_shot"] for r in rows if r["clip_zero_shot"] is not None]),
            "chance": mean([1 / len(r["classes"]) for r in rows]),
            "seconds_per_model": gen_time / max(1, len(rows) * a.samples),
        }
        report[split] = {"summary": summary, "tasks": rows}
        print(f"[{split}] " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                         for k, v in summary.items()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"arch": arch, "num_params": tok.num_params, **report}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
''',
    "t2v/llm.py": r'''"""Free-form request -> list of class names, using an OpenAI-compatible endpoint (AMD TokenFactory).

Configure with environment variables (never commit the key):
    TOKENFACTORY_API_KEY   your key
    TOKENFACTORY_BASE_URL  e.g. https://<tokenfactory-host>/v1  (copy it from the TokenFactory page)
    TOKENFACTORY_MODEL     a chat model id listed by the endpoint
Without them, the prompt is split on commas / "and" / "و" instead.
"""
import json
import os
import re

import requests

SYSTEM = (
    "You convert a request for an image classifier into the classes it must recognise. "
    "Reply with JSON only, no prose: {\"classes\": [\"...\"], \"task\": \"one short sentence\"}. "
    "Rules: between 2 and {k} classes; each class is a short singular English noun phrase (1-3 words) naming "
    "something visually recognisable in a photo; class names are ALWAYS written in English, even when the request is in Arabic or another language; "
    "no duplicates; keep the user's order when they list classes."
)


def _fallback(prompt, k):
    parts = re.split(r"\s*(?:,|،|;|\n|\band\b|\bor\b|\s+و\s+|\s+أو\s+)\s*", prompt)
    names = []
    for part in parts:
        part = part.strip(" .:-").lower()
        if part and part not in names:
            names.append(part)
    return names[:k]


def parse_classes(prompt, max_classes=10, timeout=60):
    key = os.environ.get("TOKENFACTORY_API_KEY")
    base = os.environ.get("TOKENFACTORY_BASE_URL")
    model = os.environ.get("TOKENFACTORY_MODEL")
    if not (key and base and model):
        return _fallback(prompt, max_classes), {"source": "fallback"}
    r = requests.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": SYSTEM.replace("{k}", str(max_classes))},
                           {"role": "user", "content": f"Request: {prompt}\n\nAnswer with English class names."}]},
        timeout=timeout,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0) if match else text)
    names = []
    for c in data.get("classes", []):
        c = str(c).strip().lower()
        if c and c not in names:
            names.append(c)
    return names[:max_classes], {"source": model, "task": data.get("task", "")}
''',
    "t2v/features.py": r'''"""Frozen CLIP image features for CIFAR-100, computed once (python -m t2v.features).

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
''',
}
os.makedirs("/content/t2v", exist_ok=True)
os.chdir("/content")
for path, src in FILES.items():
    with open(path, "w") as f:
        f.write(src)
print(f"updated {len(FILES)} files")

if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
ZOO, CKPT, py = f"{OUT}/zoo", f"{OUT}/t2v_resnet20_{VERSION}.pt", sys.executable

sh(f"{py} -m t2v.train_gen --zoo {ZOO} --out {CKPT} --token_size 1024 --d_model 512 --denoiser mlp --mlp_hidden 1024 "
   f"--bs 32 --tokens_per_step 4096 --steps {GEN_STEPS} --warmup 300 --pred x0 --norm element --head_weight 10 --joint_layers 4 --fp16 "
   f"--prompt_ensemble --emb_noise 0.02 --ema 0.999 --cosine --log_every 250 --save_every 500")
sh(f"{py} -m t2v.evaluate --ckpt {CKPT} --zoo {ZOO} --max_tasks 30 --samples 1 --steps 20 --out {OUT}/eval_{VERSION}.json")

r = json.load(open(f"{OUT}/eval_{VERSION}.json"))
print("\n" + "=" * 72)
print(f"T2V on ResNet-20: {r['num_params']:,} parameters generated per model")
for split, name in (("eval_seen", "seen classes"), ("eval_unseen", "half unseen"), ("eval_novel", "all unseen")):
    if split in r:
        s = r[split]["summary"]
        print(f"{name:13s} generated {s['generated_mean']:.3f} | oracle {s['oracle']:.3f} | "
              f"selection {s['selection']:.3f} | chance {s['chance']:.3f} | {s['seconds_per_model']:.2f} s/model")
print("=" * 72)
