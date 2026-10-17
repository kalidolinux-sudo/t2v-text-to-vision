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
from .data import dataset_from_meta, prep
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
    p.add_argument("--splits", default="eval_seen,eval_novel,eval_novel_hard")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--body_noise", type=float, default=1.0, help="0 = start the body at the mean when sampling")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/diagnose.json")
    a = p.parse_args()
    dev = pick_device(a.device)
    torch.manual_seed(0)
    random.seed(0)

    gen, tok, ck = load_generator(a.ckpt, dev)
    arch = ck["arch"]
    with open(f"{a.zoo}/meta.json") as f:
        meta = json.load(f)
    data = dataset_from_meta(meta, a.data, dev)
    E = embed_names(ck["names"], dev, ensemble=ck.get("prompt_ensemble", False), descriptions=ck.get("descriptions"))
    seen = meta["seen"]
    net = target.build(arch).to(dev)
    nb = sum(torch.Size(s).numel() for n, s in tok.spec if n not in target.HEAD)   # body/head boundary
    K = tok.K

    def gen_flat(cls, emb=None, n=1):
        return generate_flat(gen, tok, cls, n=n, steps=a.steps, emb=E[cls] if emb is None else emb, amp=a.fp16,
                             body_noise=a.body_noise)

    def acc(flat, cls):
        xt, yt = data.subset(cls, train=False)
        return accuracy(target.load_flat(net, flat.clone()), xt, yt, k=len(cls))

    splits = {}
    for split in a.splits.split(","):
        try:
            flat, classes, oracle = load_split(a.zoo, split)
        except SystemExit:
            continue
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
            other = random.sample([c for c in range(len(meta["names"])) if c not in cls], len(cls))
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

        # 6. memorisation, on the classifier rows only: body tokens are mostly fine-tuning noise, so a
        # whole-network distance says nothing about copying. Each generated row is compared with the right
        # row (oracle) and with the nearest row of any training model.
        head_pos = tok.head_pos
        train_rows = train_z[:, head_pos][..., :tok.valid[head_pos[0]].sum()]          # (N, K, F+1)
        train_rows = train_rows[train_cls >= 0]                                        # real rows only
        d_or, d_nn = [], []
        for zg, zo, (_, cls, _) in zip(z_g, z_o, tasks):
            F1 = train_rows.shape[-1]
            g_rows, o_rows = zg[head_pos[:len(cls)], :F1], zo[head_pos[:len(cls)], :F1]
            d_or += (g_rows - o_rows).norm(dim=-1).tolist()
            d_nn += torch.cdist(g_rows, train_rows).min(1).values.tolist()
        r["6_memorisation_rows"] = {"row_distance_to_oracle": mean(d_or),
                                    "row_distance_to_nearest_train_row": mean(d_nn),
                                    "rows_closer_to_a_train_row": mean([n < o for n, o in zip(d_nn, d_or)])}

        # body = zoo mean? compare with the error of simply outputting the zoo's average model
        z_m = tok.encode(mean_flat[None]).expand_as(z_o)
        err_m = ((z_m - z_o).abs() * tok.valid).sum(-1) / tok.valid.sum(-1).clamp_min(1)
        body = tok.slots < 0
        r["4b_body_vs_mean_model"] = {"generated_body_error": err[:, body].mean().item(),
                                      "zoo_mean_body_error": err_m[:, body].mean().item()}

        # 7. per-class recall vs CLIP similarity to the nearest seen class
        if split.startswith("eval_novel"):
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
        # 8. semantic head prior (DeViSE-style): a ridge map from class text embeddings to the generic model's
        # classifier rows, fitted on seen classes only; lambda chosen by 5-fold CV over seen classes (row cosine),
        # never on test tasks. Rows go on the zoo-mean body and on the generated body.
        if "ridge_W" not in globals():
            g_state = torch.load(f"{a.zoo}/generic.pt", map_location=dev)["state"]
            rows_all = torch.cat([g_state["fc.weight"], g_state["fc.bias"][:, None]], 1)
            X, Y = E[seen], rows_all[seen]
            folds = torch.arange(len(seen)) % 5
            best = None
            for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
                cos = []
                for f in range(5):
                    tr_, te_ = folds != f, folds == f
                    W = torch.linalg.solve(X[tr_].T @ X[tr_] + lam * torch.eye(X.shape[1], device=dev), X[tr_].T @ Y[tr_])
                    cos.append(torch.nn.functional.cosine_similarity(X[te_] @ W, Y[te_], dim=-1).mean().item())
                if best is None or mean(cos) > best[1]:
                    best = (lam, mean(cos))
            lam = best[0]
            globals()["ridge_W"] = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=dev), X.T @ Y)
            globals()["ridge_info"] = {"lambda (5-fold CV on seen classes)": lam, "cv_row_cosine": best[1]}

        def with_rows(base, cls):
            f = base.clone()
            w0 = nb
            F = (f.numel() - nb - K) // K
            rows = E[cls] @ ridge_W
            w = torch.zeros(K, F, device=dev)
            b = torch.zeros(K, device=dev)
            w[:len(cls)], b[:len(cls)] = rows[:, :F], rows[:, F]
            f[w0:w0 + K * F] = w.flatten()
            f[w0 + K * F:] = b
            return f
        r["8_semantic_head_prior"] = {**ridge_info,
                                      "ridge head + zoo-mean body": mean([acc(with_rows(mean_flat, c), c) for _, c, _ in tasks]),
                                      "ridge head + generated body": mean([acc(with_rows(g, c), c) for g, (_, c, _) in zip(g_all, tasks)]),
                                      "T2V (generated head + body)": mean(right)}
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
