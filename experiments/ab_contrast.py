"""A/B: stored TokenFactory descriptions vs + contrastive descriptions, on 10-way tasks of never-seen classes,
evaluated on real ImageNet validation photos (50 per class). Same tasks, same generator, only the condition differs."""
import sys, json, random, time
import os; R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, R)
import torch
from t2v.generate import load_generator, generate_flat, build_model
from t2v.text_embed import embed_names
from t2v.contrast import refine
from t2v import tokenfactory


N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
POOL = sys.argv[2] if len(sys.argv) > 2 else "all"
torch.manual_seed(0)
gen, tok, ck = load_generator(f"{R}/checkpoints/t2v_1k.pt", "cpu")
split = json.load(open(f"{R}/data/imagenet_split.json"))
names = ck["names"]
pool_ids = {"all": split["holdout_random"] + split["holdout_hard"], "hard": split["holdout_hard"], "random": split["holdout_random"]}[POOL]
val = torch.load(f"{R}/data/imagenet32_val.pt")
rng = random.Random(1234)
res = []
for t in range(N_TASKS):
    ids = rng.sample(pool_ids, 10)
    classes = [names[i] for i in ids]
    descs = {c: gen.descriptions.get(c) for c in classes}
    m = torch.isin(val["y"], torch.tensor(ids)); x, y = val["x"][m], val["y"][m]
    yl = torch.tensor([ids.index(int(v)) for v in y])
    def acc(d):
        torch.manual_seed(0)
        flat = generate_flat(gen, tok, classes, steps=5, body_noise=0.0, gate="auto", descriptions=d)[0]
        model = build_model(ck["arch"], flat, len(classes)).eval()
        with torch.no_grad():
            p = model(x.float().div(255)).argmax(1)
        return (p == yl).float().mean().item()
    a0 = acc(descs)
    emb = embed_names(classes, "cpu", ensemble=gen.prompt_ensemble, descriptions=descs)
    known = ((emb @ gen.seen_emb.T).max(1).values >= gen.gate).tolist()
    d2, pairs, extra = refine(classes, descs, emb, [not k for k in known], k=3, n=2)
    a1 = acc(d2) if extra else a0
    res.append((a0, a1, len(extra)))
    print(f"task {t:2d}  base {a0:.3f}  +contrast {a1:.3f}  diff {a1 - a0:+.3f}  pairs {[(a, b) for a, b, _ in pairs]}", flush=True)
b = sum(r[0] for r in res) / len(res); c = sum(r[1] for r in res) / len(res)
d = torch.tensor([r[1] - r[0] for r in res])
se = d.std().item() / len(res) ** 0.5
wins = int((d > 0).sum()); losses = int((d < 0).sum())
print(f"\nPOOL={POOL} tasks={len(res)}  base {b:.4f}  +contrast {c:.4f}  diff {c - b:+.4f} ± {se:.4f} (s.e.)  wins {wins} losses {losses}")
print(f"TokenFactory calls {tokenfactory.STATS['calls']}, {tokenfactory.STATS['seconds']:.0f} s")
