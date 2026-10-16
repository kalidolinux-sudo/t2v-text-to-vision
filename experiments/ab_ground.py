"""A/B: requests written with WordNet synonyms instead of our class names ("tabby cat", "orca", ...).
base     = the product with a key but without grounding: synonym names, live TokenFactory descriptions, live analogies
grounded = the agent first maps each synonym onto the vocabulary (then stored condition, learned rows)
Evaluated on real ImageNet validation photos (50 per class), same tasks for both."""
import sys, json, random
import os; R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, R)
import torch
from t2v.generate import load_generator, generate_flat, build_model
from t2v.describe import describe
from t2v.text_embed import embed_names
from t2v import neighbors as N, tokenfactory
from t2v.ground import ground


N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
gen, tok, ck = load_generator(f"{R}/checkpoints/t2v_1k.pt", "cpu")
names = ck["names"]
full = json.load(open(f"{R}/data/imagenet_names.json"))["full"]
seen = set(gen.seen_ids)
syn = {}
for i, (f, s) in enumerate(zip(full, names)):
    alts = [a.strip() for a in f.split(",") if a.strip() and a.strip().lower() != s.lower() and s.lower() not in a.strip().lower() and a.strip().lower() not in s.lower()]
    if alts:
        syn[i] = alts[0]
print("classes with a synonym:", len(syn), "seen among them:", sum(i in seen for i in syn))
rng = random.Random(7)
pool = list(syn)
tasks = [rng.sample(pool, 10) for _ in range(N_TASKS)]
all_syn = sorted({syn[i] for t in tasks for i in t})
print("synonym names:", len(all_syn))
# live TokenFactory work for the baseline, batched
descs = describe(all_syn)
print("described", len(descs))
mapping = ground(all_syn, names)
print("grounded", sum(v is not None for v in mapping.values()), "of", len(all_syn))
correct = sum(1 for t in tasks for i in t if mapping.get(syn[i]) == names[i]); total = sum(len(t) for t in tasks)
wrong = sorted({(syn[i], mapping[syn[i]], names[i]) for t in tasks for i in t if mapping.get(syn[i]) not in (None, names[i])})
print(f"grounding: {correct}/{total} task-classes mapped to the right class; wrong maps: {wrong[:10]}")
# baseline analogies for synonyms (the product asks them live for names it has not seen)
store = gen.neighbors["store"]
seen_names = [names[i] for i in gen.seen_ids]
emb_s = embed_names(all_syn, "cpu", ensemble=gen.prompt_ensemble, descriptions=descs)
known_s = ((emb_s @ gen.seen_emb.T).max(1).values >= gen.gate).tolist()
need = [n for n, k in zip(all_syn, known_s) if not k and n not in store]
cands = N.shortlist(emb_s[[all_syn.index(n) for n in need]], gen.seen_emb, seen_names, 20) if need else []
store.update(N.ask(need, dict(zip(need, cands))))
print("base: synonyms the gate calls known:", sum(known_s), "of", len(all_syn), "| analogies asked:", len(need))
val = torch.load(f"{R}/data/imagenet32_val.pt")
res = []
for t, ids in enumerate(tasks):
    m = torch.isin(val["y"], torch.tensor(ids)); x, y = val["x"][m], val["y"][m]
    yl = torch.tensor([ids.index(int(v)) for v in y])
    def acc(classes, d):
        flat = generate_flat(gen, tok, classes, steps=5, body_noise=0.0, gate="auto", descriptions=d)[0]
        model = build_model(ck["arch"], flat, len(classes)).eval()
        with torch.no_grad():
            return (model(x.float().div(255)).argmax(1) == yl).float().mean().item()
    base_names = [syn[i] for i in ids]
    a0 = acc(base_names, descs)
    g_names = [mapping.get(n) or n for n in base_names]
    a1 = acc(g_names, descs)
    res.append((a0, a1)); print(f"task {t:2d}  base {a0:.3f}  grounded {a1:.3f}  diff {a1 - a0:+.3f}", flush=True)
d = torch.tensor([b - a for a, b in res])
print(f"\ntasks={len(res)}  base {sum(r[0] for r in res)/len(res):.4f}  grounded {sum(r[1] for r in res)/len(res):.4f}  "
      f"diff {d.mean():+.4f} ± {d.std().item()/len(res)**0.5:.4f} (s.e.)  wins {(d>0).sum()} losses {(d<0).sum()}")
print(f"TokenFactory calls {tokenfactory.STATS['calls']}, {tokenfactory.STATS['seconds']:.0f} s")
