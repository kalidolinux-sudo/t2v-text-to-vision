"""Build the model zoo the generator learns from (Tina's recipe).

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
