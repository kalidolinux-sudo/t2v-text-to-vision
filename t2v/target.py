"""Vision models whose weights get generated. Every model ends in a Linear named `fc`."""
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


def _head_offsets(arch):
    o, offs = 0, {}
    for name, shape in param_spec(build(arch)):
        offs[name] = (o, shape)
        o += int(torch.Size(shape).numel())
    (w0, (K, feat)), (b0, _) = offs[HEAD[0]], offs[HEAD[1]]
    return w0, b0, K, feat


def head_rows(arch, flat):
    """flat (..., P) -> classifier rows (..., K, feat + 1) as [weight row, bias]."""
    w0, b0, K, feat = _head_offsets(arch)
    w = flat[..., w0:w0 + K * feat].reshape(*flat.shape[:-1], K, feat)
    return torch.cat([w, flat[..., b0:b0 + K, None]], -1)


def add_head_rows(arch, flat, rows, sign=1.0):
    """Add (sign=+1) or remove (sign=-1) `rows` (..., k, feat + 1) to the first k classifier rows of `flat`."""
    w0, b0, K, feat = _head_offsets(arch)
    k = rows.shape[-2]
    flat = flat.clone()
    rows = rows.to(flat.device, flat.dtype)
    w = flat[..., w0:w0 + K * feat].reshape(*flat.shape[:-1], K, feat).clone()
    w[..., :k, :] += sign * rows[..., :feat]
    flat[..., w0:w0 + K * feat] = w.reshape(*flat.shape[:-1], K * feat)
    flat[..., b0:b0 + k] += sign * rows[..., feat]
    return flat


def fit_semantic_prior(emb, rows, class_ids, lams=(1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0), folds=5):
    """Ridge map from class text embeddings to classifier rows (DeViSE-style). Lambda is chosen by
    cross-validation over *classes* (rows of held-out classes are predicted from the others), so the
    choice reflects generalisation to unseen classes and never looks at test tasks.
    emb (N, E), rows (N, F+1), class_ids (N,) -> W (E, F+1), info."""
    fold = class_ids % folds
    eye = torch.eye(emb.shape[1], device=emb.device)
    best = None
    for lam in lams:
        cos = []
        for f in range(folds):
            tr, te = fold != f, fold == f
            if te.sum() == 0:
                continue
            W = torch.linalg.solve(emb[tr].T @ emb[tr] + lam * eye, emb[tr].T @ rows[tr])
            cos.append(torch.nn.functional.cosine_similarity(emb[te] @ W, rows[te], dim=-1).mean().item())
        score = sum(cos) / len(cos)
        if best is None or score > best[1]:
            best = (lam, score)
    W = torch.linalg.solve(emb.T @ emb + best[0] * eye, emb.T @ rows)
    return W, {"lambda": best[0], "cv_row_cosine_on_held_out_classes": best[1]}


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
