"""Build the model zoo the generator learns from (Tina's recipe, extended).

Stage 1  `generic`: one model trained on every seen class.
Stage 2  `pmodels`: many personalised 10-way models, each fine-tuned from the generic model on one
         task (a set of classes). Every p-model is stored as a flat parameter vector plus its class ids.
Stage 3  `analyze`: how much of the body variation across the zoo is task signal rather than noise.

Two choices make the body carry task information instead of fine-tuning noise:
  --finetune tail:N   only the last N residual blocks + classifier adapt; earlier layers stay generic
  --soup S            each task is fine-tuned S times with different seeds and the weights averaged
                      (valid because all runs start from the same generic model; Wortsman et al. 2022),
                      which cancels seed noise and leaves the task-specific part

Shards let several processes share one GPU:
    for s in 0 1 2 3; do python -m t2v.zoo pmodels --shard $s --num_shards 4 & done
"""
import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import target
from .data import cluster_classes, dataset_from_meta, load_dataset, prep, sample_task, split_classes

SPLITS = ["train", "eval_seen", "eval_unseen", "eval_novel", "eval_novel_hard"]


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
    for mod in model.modules():  # frozen BatchNorm layers keep their statistics fixed
        if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()
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


def trainable(name, finetune, arch_blocks):
    """Which parameters adapt during personalisation."""
    if finetune == "all":
        return True
    if finetune == "head":
        return name.startswith("fc.")
    if finetune.startswith("tail:"):
        n = int(finetune.split(":")[1])
        tail = [f"layers.{i}." for i in range(arch_blocks - n, arch_blocks)]
        return name.startswith("fc.") or any(name.startswith(t) for t in tail)
    raise ValueError(finetune)


def cmd_generic(a):
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    data = load_dataset(a.dataset, a.data, dev, features=target.uses_features(a.arch), per_class=a.per_class)
    n_cls = len(data.names)
    if a.split_file:  # a fixed split shipped with the code (TokenFactory's answers were computed for it)
        with open(a.split_file) as f:
            sp = json.load(f)
        assert sp["names"] == data.names, "split file was made for a different class list"
        coarse, seen, hold_rand, hold_hard = sp["coarse"], sp["seen"], sp["holdout_random"], sp["holdout_hard"]
    else:
        coarse = data.coarse
        if coarse is None or a.hard_holdout:
            from .text_embed import embed_names
            coarse = cluster_classes(embed_names(data.names, "cpu", ensemble=True), a.clusters, a.seed)
        seen, hold_rand, hold_hard = split_classes(a.num_holdout, a.seed, n_cls, a.hard_holdout, coarse)
    x, y = data.subset(seen)
    y = torch.tensor(seen, device=dev)[y]  # keep the original class ids for the n_cls-way head
    model = target.build(a.arch, n_cls).to(dev)
    t0 = time.time()
    train(model, x, y, a.epochs, a.lr, a.bs, wd=a.wd)
    xt, yt = data.subset(seen, train=False)
    acc = accuracy(model, xt, torch.tensor(seen, device=dev)[yt])
    os.makedirs(a.out, exist_ok=True)
    torch.save({"arch": a.arch, "state": model.state_dict()}, os.path.join(a.out, "generic.pt"))
    meta = {"arch": a.arch, "dataset": a.dataset, "per_class": a.per_class, "names": data.names, "coarse": coarse,
            "seen": seen, "holdout": sorted(hold_rand + hold_hard), "holdout_random": hold_rand,
            "holdout_hard": hold_hard, "max_classes": target.MAX_CLASSES,
            "num_params": sum(int(torch.Size(s).numel()) for _, s in target.param_spec(target.build(a.arch)))}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"generic {a.arch} on {a.dataset}: {acc:.3f} test acc over {len(seen)} seen classes "
          f"({len(hold_rand)} random + {len(hold_hard)} hard held out), {time.time() - t0:.0f}s")


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


def make_tasks(a, meta, K):
    """Tasks are a pure function of (split, seed, index), so shards never overlap and reruns resume."""
    seen = meta["seen"]
    hold_rand = meta.get("holdout_random") or meta["holdout"]
    hold_hard = meta.get("holdout_hard") or []
    rng = random.Random(f"{a.split}-{a.seed}")
    tasks = []
    for _ in range(a.n):
        if a.split == "eval_novel":            # every class from the random holdout
            t = rng.sample(hold_rand, K)
        elif a.split == "eval_novel_hard":     # every class from held-out semantic groups
            t = rng.sample(hold_hard, min(K, len(hold_hard)))
        elif a.split == "eval_unseen":
            n_new = max(1, round(a.unseen_frac * K))
            t = rng.sample(meta["holdout"], n_new) + rng.sample(seen, K - n_new)
            rng.shuffle(t)
        else:
            t = sample_task(rng, seen, meta["coarse"], a.min_k, K)
        tasks.append(t)
    if a.split == "eval_seen":  # unseen combinations of seen classes
        train_rng = random.Random(f"train-{a.seed}")
        train_sets = {frozenset(sample_task(train_rng, seen, meta["coarse"], a.min_k, K)) for _ in range(a.n_train_ref)}
        tasks = [t for t in tasks if frozenset(t) not in train_sets]
    return tasks


def cmd_pmodels(a):
    dev = pick_device(a.device)
    with open(os.path.join(a.out, "meta.json")) as f:
        meta = json.load(f)
    g = torch.load(os.path.join(a.out, "generic.pt"), map_location=dev)
    arch, K = g["arch"], meta["max_classes"]
    data = dataset_from_meta(meta, a.data, dev, features=target.uses_features(arch))
    text_emb = None
    if target.has_prior(arch):
        from .text_embed import embed_names
        text_emb = embed_names(meta["names"], dev, ensemble=True)
    n_blocks = len(getattr(target.build(arch), "layers", []))
    tasks = make_tasks(a, meta, K)

    path = os.path.join(a.out, f"{a.split}_shard{a.shard}.pt")
    done = torch.load(path) if os.path.exists(path) else {"flat": [], "classes": [], "acc": [], "index": [], "seed_sqdist": []}
    done.setdefault("seed_sqdist", [])
    have = set(done["index"])
    mine = [i for i in range(len(tasks)) if i % a.num_shards == a.shard and i not in have]
    print(f"{a.split} shard {a.shard}: {len(mine)} p-models to train ({len(have)} already done), "
          f"finetune={a.finetune}, soup={a.soup}")
    t0 = time.time()
    for n, i in enumerate(mine):
        classes = tasks[i]
        x, y = data.subset(classes)
        runs = []
        for s in range(a.soup):
            torch.manual_seed(a.seed * 1_000_003 + SPLITS.index(a.split) * 100_003 + i + 7919 * s)
            m = personalise(g["state"], arch, classes, dev, text_emb)
            for name, p in m.named_parameters():
                p.requires_grad_(trainable(name, a.finetune, n_blocks))
            train(m, x, y, a.epochs, a.lr, a.bs, k=len(classes), wd=a.wd)
            runs.append(target.flatten(m))
        flat = torch.stack(runs).mean(0)  # model soup: seed noise averages out
        if a.soup > 1:
            done["seed_sqdist"].append(((runs[0] - runs[1]) ** 2).sum().item())
        m = target.load_flat(target.build(arch).to(dev), flat.to(dev))
        xt, yt = data.subset(classes, train=False)
        done["acc"].append(accuracy(m, xt, yt, k=len(classes)))
        if text_emb is not None:  # store the correction to the zero-shot head, not the raw head
            flat = target.add_head_prior(arch, flat, text_emb[classes].cpu(), sign=-1.0)
        done["flat"].append(flat.half() if a.fp16 else flat)
        done["classes"].append(classes + [-1] * (K - len(classes)))
        done["index"].append(i)
        if (n + 1) % a.save_every == 0 or n + 1 == len(mine):
            torch.save(done, path)
            rate = (n + 1) / (time.time() - t0)
            print(f"  {n + 1}/{len(mine)}  mean acc {sum(done['acc']) / len(done['acc']):.3f}  {rate:.2f} tasks/s", flush=True)


def cmd_analyze(a):
    """Signal-to-noise of the body across the zoo (needs --soup 2 or more).
    Each stored model is a soup of S runs: Var_tasks(soup) = Var_signal + Var_noise / S, and the
    distance between two runs of the same task gives Var_noise = E[(run1 - run2)^2] / 2."""
    with open(os.path.join(a.out, "meta.json")) as f:
        meta = json.load(f)
    flats, sq = [], []
    for p in sorted(glob.glob(os.path.join(a.out, "train_shard*.pt"))):
        s = torch.load(p)
        flats += s["flat"]
        sq += s.get("seed_sqdist", [])
    if not sq:
        raise SystemExit("no seed distances stored (build the zoo with --soup 2 or more)")
    flat = torch.stack(flats).float()
    spec = target.param_spec(target.build(meta["arch"]))
    var_tasks = flat.var(0)                           # per element
    moving = var_tasks > 0                            # frozen layers do not vary at all
    n_moving = int(moving.sum())
    var_noise = sum(sq) / len(sq) / 2 / max(1, n_moving)
    var_signal = var_tasks[moving].mean().item() - var_noise / a.soup
    report = {"moving_params": n_moving, "frozen_params": int((~moving).sum()),
              "noise_var_per_param": var_noise, "signal_var_per_param": var_signal,
              "signal_to_noise": var_signal / max(var_noise, 1e-30)}
    o, per_layer = 0, {}
    for name, shape in spec:
        nel = int(torch.Size(shape).numel())
        if moving[o:o + nel].any():
            per_layer[name] = var_tasks[o:o + nel].mean().item()
        o += nel
    print(json.dumps(report, indent=1))
    print("varying tensors (task-to-task variance):", {k: round(v, 6) for k, v in list(per_layer.items())[:12]})
    with open(os.path.join(a.out, "zoo_snr.json"), "w") as f:
        json.dump(report, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generic", "pmodels", "analyze"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="zoo")
        s.add_argument("--data", default="data")
        s.add_argument("--device", default="auto")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--bs", type=int, default=128)
    g = sub.choices["generic"]
    g.add_argument("--arch", default="resnet20", choices=list(target.ARCHS))
    g.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet32"])
    g.add_argument("--per_class", type=int, default=260, help="imagenet32: training images per class")
    g.add_argument("--num_holdout", type=int, default=20, help="classes never used by the zoo")
    g.add_argument("--hard_holdout", type=int, default=0,
                   help="how many of the held-out classes are whole semantic clusters (the rest are random)")
    g.add_argument("--clusters", type=int, default=50, help="semantic clusters (CLIP k-means) for sampling/holdout")
    g.add_argument("--split_file", help="JSON with names/coarse/seen/holdout_random/holdout_hard to use as is")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--wd", type=float, default=5e-4)
    m = sub.choices["pmodels"]
    m.add_argument("--split", default="train", choices=SPLITS)
    m.add_argument("--n", type=int, default=1000, help="number of tasks for this split")
    m.add_argument("--n_train_ref", type=int, default=1000, help="train-split size, to exclude its tasks from eval_seen")
    m.add_argument("--min_k", type=int, default=10, help="smallest class count per task (Tina pads to 10)")
    m.add_argument("--unseen_frac", type=float, default=0.5, help="eval_unseen: fraction of held-out classes per task")
    m.add_argument("--epochs", type=int, default=10)
    m.add_argument("--lr", type=float, default=0.01)
    m.add_argument("--wd", type=float, default=5e-4)
    m.add_argument("--finetune", default="all", help="all | head | tail:N (last N residual blocks + classifier)")
    m.add_argument("--soup", type=int, default=1, help="fine-tune each task this many times and average the weights")
    m.add_argument("--fp16", action="store_true", help="store p-models in fp16 (halves zoo size)")
    m.add_argument("--shard", type=int, default=0)
    m.add_argument("--num_shards", type=int, default=1)
    m.add_argument("--save_every", type=int, default=25)
    z = sub.choices["analyze"]
    z.add_argument("--soup", type=int, default=2)
    a = p.parse_args()
    {"generic": cmd_generic, "pmodels": cmd_pmodels, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
