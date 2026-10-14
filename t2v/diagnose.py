"""Inference-only diagnostics of a trained T2V generator: where does it fail, and why?

    python -m t2v.diagnose --ckpt ckpt/generator.pt --zoo zoo

Seven probes, a few minutes on one GPU, no training:
  1 conditioning   does the output depend on the classes? (right vs wrong class embeddings)
  2 equivariance   does permuting the class order permute the classifier rows?
  3 body vs head   which half carries the error? (swap generated and fine-tuned halves)
  4 layer errors   which layers are generated worst?
  5 diversity      do different noise draws give different models, or one mode?
  6 memorisation   is the output closer to some training model than to the right answer?
  7 unseen classes does accuracy on a new class follow its CLIP similarity to the seen classes?
"""
import argparse
import json
import random

import torch

from . import target
from .data import Cifar100, prep
from .generate import generate_flat, load_generator
from .text_embed import embed_names
from .train_gen import load_split
from .zoo import accuracy, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--zoo", default="zoo")
    p.add_argument("--data", default="data")
    p.add_argument("--tasks", type=int, default=20, help="tasks per split")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/diagnose.json")
    a = p.parse_args()
    dev = pick_device(a.device)
    torch.manual_seed(0)
    random.seed(0)

    gen, tok, ck = load_generator(a.ckpt, dev)
    arch = ck["arch"]
    data = Cifar100(a.data, dev, features=target.uses_features(arch))
    E = embed_names(ck["names"], dev, ensemble=ck.get("prompt_ensemble", False), descriptions=ck.get("descriptions"))
    with open(f"{a.zoo}/meta.json") as f:
        meta = json.load(f)
    seen = meta["seen"]
    net = target.build(arch).to(dev)
    nb = sum(torch.Size(s).numel() for n, s in tok.spec if n not in target.HEAD)   # body/head boundary
    K = tok.K

    def gen_flat(cls, emb=None, n=1):
        return generate_flat(gen, tok, cls, n=n, steps=a.steps, emb=E[cls] if emb is None else emb, amp=a.fp16)

    def acc(flat, cls):
        xt, yt = data.subset(cls, train=False)
        return accuracy(target.load_flat(net, flat.clone()), xt, yt, k=len(cls))

    splits = {}
    for split in ("eval_seen", "eval_novel"):
        flat, classes, oracle = load_split(a.zoo, split)
        splits[split] = [(flat[i].to(dev), [c for c in classes[i].tolist() if c >= 0], oracle[i].item())
                         for i in range(min(a.tasks, len(flat)))]
    train_flat, train_cls, _ = load_split(a.zoo, "train")
    train_z = tok.encode(train_flat.to(dev))                                    # (N, T, D)
    mean_flat = train_flat.mean(0).to(dev)                                       # the zoo's average model
    report = {}
    mean = lambda v: sum(v) / max(1, len(v))

    print(f"generator {a.ckpt} | {arch} | {tok.num_params:,} params | {a.steps} steps\n")
    for split, tasks in splits.items():
        r = {}
        # 1 + 3 + 6 + 7 share the generated models
        g_all = [gen_flat(cls)[0] for _, cls, _ in tasks]

        # 1. conditioning: right classes vs embeddings of random other classes
        right = [acc(g, cls) for g, (_, cls, _) in zip(g_all, tasks)]
        wrong = []
        for _, cls, _ in tasks:
            other = random.sample([c for c in range(100) if c not in cls], len(cls))
            wrong.append(acc(gen_flat(cls, emb=E[other])[0], cls))
        r["1_conditioning"] = {"right_classes": mean(right), "wrong_classes": mean(wrong),
                               "chance": mean([1 / len(c) for _, c, _ in tasks])}

        # 2. equivariance: generate with the class order reversed, evaluate with the reversed labels
        perm = [acc(gen_flat(cls[::-1])[0], cls[::-1]) for _, cls, _ in tasks]
        r["2_equivariance"] = {"original_order": mean(right), "reversed_order": mean(perm)}

        # 3. body vs head
        def swap(body, head):
            f = head.clone()
            f[:nb] = body[:nb]
            return f
        r["3_body_vs_head"] = {
            "fine-tuned (oracle)": mean([o for _, _, o in tasks]),
            "generated body + generated head": mean(right),
            "generated body + oracle head": mean([acc(swap(g, o), c) for g, (o, c, _) in zip(g_all, tasks)]),
            "oracle body + generated head": mean([acc(swap(o, g), c) for g, (o, c, _) in zip(g_all, tasks)]),
            "zoo-mean body + generated head": mean([acc(swap(mean_flat, g), c) for g, (_, c, _) in zip(g_all, tasks)]),
        }

        # 4. per-layer error in normalised units (body layers + head)
        z_g = tok.encode(torch.stack(g_all))
        z_o = tok.encode(torch.stack([o for o, _, _ in tasks]))
        err = ((z_g - z_o).abs() * tok.valid).sum(-1) / tok.valid.sum(-1).clamp_min(1)   # (n, T)
        names = [n for n, _ in tok.spec if n not in target.HEAD] + ["fc (classifier rows)"]
        per_layer = {}
        for li in tok.layer_ids.unique().tolist():
            m = tok.layer_ids == li
            per_layer[names[li]] = err[:, m].mean().item()
        worst = sorted(per_layer.items(), key=lambda kv: -kv[1])[:5]
        r["4_layer_errors"] = {"mean": err.mean().item(), "worst_5": dict(worst)}

        # 5. diversity: 4 noise draws per task
        within, accs_spread = [], []
        for _, cls, _ in tasks[:8]:
            s = gen_flat(cls, n=4)
            within.append(tok.encode(s).std(0).mean().item())
            av = [acc(f, cls) for f in s]
            accs_spread.append(max(av) - min(av))
        across = z_g.std(0).mean().item()
        r["5_diversity"] = {"spread_across_noise": mean(within), "spread_across_tasks": across,
                            "accuracy_range_across_noise": mean(accs_spread)}

        # 6. memorisation: distance to the right answer vs to the nearest training model
        d_oracle, d_nn, shared = [], [], []
        for zg, zo, (_, cls, _) in zip(z_g, z_o, tasks):
            d_train = ((train_z - zg) ** 2 * tok.valid).sum((1, 2))
            j = d_train.argmin().item()
            d_oracle.append(((zg - zo) ** 2 * tok.valid).sum().sqrt().item())
            d_nn.append(d_train[j].sqrt().item())
            shared.append(len(set(cls) & set(c for c in train_cls[j].tolist() if c >= 0)) / len(cls))
        r["6_memorisation"] = {"distance_to_oracle": mean(d_oracle), "distance_to_nearest_train_model": mean(d_nn),
                               "closer_to_a_train_model_than_to_oracle": mean([n < o for n, o in zip(d_nn, d_oracle)]),
                               "class_overlap_with_that_train_model": mean(shared)}

        # 7. per-class recall vs CLIP similarity to the nearest seen class
        if split == "eval_novel":
            rec, sim = [], []
            seen_e = E[seen]
            for g, (_, cls, _) in zip(g_all, tasks):
                xt, yt = data.subset(cls, train=False)
                m = target.load_flat(net, g.clone()).eval()
                with torch.no_grad():
                    pred = m(prep(xt))[:, :len(cls)].argmax(1)
                for j, c in enumerate(cls):
                    rec.append((pred[yt == j] == j).float().mean().item())
                    sim.append((E[c] @ seen_e.T).max().item())
            rt, st = torch.tensor(rec), torch.tensor(sim)
            corr = torch.corrcoef(torch.stack([rt, st]))[0, 1].item()
            lo = st < st.median()
            r["7_unseen_classes"] = {"recall_vs_similarity_correlation": corr,
                                     "recall_far_from_seen (lower half)": rt[lo].mean().item(),
                                     "recall_close_to_seen (upper half)": rt[~lo].mean().item()}
        report[split] = r

        print(f"=== {split} ({len(tasks)} tasks) ===")
        for test, vals in r.items():
            print(f"  {test}")
            for k, v in vals.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        print(f"      {kk:38s} {vv:.3f}")
                else:
                    print(f"      {k:42s} {v:.3f}")
        print()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
