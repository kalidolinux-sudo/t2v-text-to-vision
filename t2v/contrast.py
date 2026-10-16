"""A tried TokenFactory step, not used by the product: contrastive descriptions for look-alike classes.

The generic descriptions say what a class looks like; they do not say how it differs from the class it is
most easily confused with in *this* request (a hammerhead next to a tiger shark, a cello next to a violin).
Here the agent gets the most similar pairs of new classes (by CLIP similarity of their current condition)
and writes, for each side, what tells it apart from the other. These lines are added to the class's
descriptions before the CLIP condition is computed, so they move the text prior of each row away from its
look-alike. Only classes the known-class gate calls new are refined; known classes keep their exact
training-time condition.

Measured with experiments/ab_contrast.py (30 tasks of 10 never-seen classes, real ImageNet photos):
+0.03 ± 0.07 points, 11 wins and 11 losses. No gain, so the app and the notebook do not use it.
"""
import json
import re

import torch

from . import tokenfactory

SYSTEM = (
    "You help a small image classifier tell look-alike classes apart in 32x32 photos. For each pair you get, "
    "write {n} short visual descriptions (8-15 words) for EACH of the two classes, stating what distinguishes "
    "it from the other class of the pair: shape, colour, texture, parts, typical setting. Reply with JSON only: "
    "{{\"<class>\": [\"...\", ...], ...}} using the class names exactly as given."
)


def confusable_pairs(names, emb, candidates, k=3, min_sim=0.0):
    """The k most CLIP-similar pairs among `candidates` (indices into names/emb), most similar first."""
    sims = emb @ emb.T
    pairs = [(sims[i, j].item(), i, j) for a, i in enumerate(candidates) for j in candidates[a + 1:]]
    pairs = [p for p in sorted(pairs, reverse=True) if p[0] >= min_sim][:k]
    return [(names[i], names[j], s) for s, i, j in pairs]


def contrastive_descriptions(pairs, n=2, timeout=120):
    """{class: [contrastive descriptions]} for every class in `pairs`, from one TokenFactory call.
    Returns {} without a key or when the reply cannot be used."""
    if not pairs:
        return {}
    prompt = "Pairs:\n" + "\n".join(f"- {a} vs {b}" for a, b, _ in pairs)
    try:
        text = tokenfactory.chat([{"role": "system", "content": SYSTEM.format(n=n)},
                                  {"role": "user", "content": prompt}], temperature=0.3, max_tokens=2000,
                                 timeout=timeout)
    except Exception as e:  # a failed call must never block generation
        print(f"contrast: request failed ({e})")
        return {}
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0) if m else text)
    except json.JSONDecodeError:
        return {}
    wanted = {c.lower(): c for a, b, _ in pairs for c in (a, b)}
    out = {}
    for k, v in data.items():
        c = wanted.get(str(k).strip().lower())
        if c and isinstance(v, list):
            out.setdefault(c, []).extend(str(d).strip() for d in v if str(d).strip())
    return out


@torch.no_grad()
def refine(names, descs, emb, is_new, k=3, n=2):
    """Add contrastive descriptions to the new classes of the k most confusable new-new pairs.
    Returns (descs with the additions, the pairs used, {class: added lines})."""
    cand = [i for i, nw in enumerate(is_new) if nw]
    if len(cand) < 2:
        return descs, [], {}
    pairs = confusable_pairs(names, emb, cand, k)
    extra = contrastive_descriptions(pairs, n)
    merged = {c: list(descs.get(c) or []) + extra.get(c, []) for c in names}
    return merged, pairs, extra
