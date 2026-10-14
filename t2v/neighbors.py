"""TokenFactory role 3: visual analogies for classes the generator has never seen.

The diagnostics showed that for a new class the generator snaps its classifier row to some known class.
Here an LLM on AMD TokenFactory reasons about it explicitly: from a CLIP shortlist of known classes it
picks the few that look most alike in a photo, with weights ("zebra: 0.6 horse, 0.4 tiger"). The row
prior for the new class is then the weighted mix of those classes' learned rows, blended with the ridge
prior by a weight chosen with cross-validation on known classes (see train_gen).

    python -m t2v.neighbors --zoo zoo --descriptions data/descriptions.json --out data/neighbors.json
"""
import argparse
import json
import os
import re

import requests
import torch

SYSTEM = (
    "You are a computer-vision expert. For each target class you get a list of candidate classes. Pick "
    "the 1 to 3 candidates that look most similar to the target in a small photo (overall shape, texture, "
    "colour, parts, typical background), and give each a weight; weights sum to 1 and a larger weight means "
    "more similar. Use only candidate names, copied exactly. Reply with JSON only: "
    "{\"<target>\": {\"<candidate>\": <weight>, ...}, ...}"
)


def _chat(messages, timeout=180):
    key, base, model = (os.environ.get(k) for k in ("TOKENFACTORY_API_KEY", "TOKENFACTORY_BASE_URL", "TOKENFACTORY_MODEL"))
    if not (key and base and model):
        return None
    r = requests.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0, "max_tokens": 3000,
                            "response_format": {"type": "json_object"}, "messages": messages}, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def shortlist(target_emb, cand_emb, cand_names, k=20, exclude=None):
    """The k most CLIP-similar candidate names for each target embedding."""
    sims = target_emb @ cand_emb.T
    out = []
    for i in range(len(target_emb)):
        order = sims[i].argsort(descending=True).tolist()
        names = [cand_names[j] for j in order if not exclude or cand_names[j] != exclude[i]]
        out.append(names[:k])
    return out


def ask(targets, candidates, batch=5, cache=None, retries=2):
    """{target: {candidate: weight}} for every target the LLM answered validly. Answers naming anything
    outside the target's candidate list are dropped; weights are renormalised."""
    out = dict(cache or {})
    todo = [t for t in targets if t not in out]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        prompt = "\n".join(f"Target: {t}\nCandidates: {', '.join(candidates[t])}" for t in chunk)
        data = None
        for _ in range(retries + 1):
            try:
                text = _chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            except requests.RequestException:
                continue
            if text is None:
                return out
            m = re.search(r"\{.*\}", text, re.S)
            try:
                data = json.loads(m.group(0) if m else text)
                break
            except json.JSONDecodeError:
                continue
        if not data:
            continue
        for t in chunk:
            picks = data.get(t) or data.get(t.lower()) or {}
            allowed = {c.lower(): c for c in candidates[t]}
            clean = {}
            for name, w in (picks.items() if isinstance(picks, dict) else []):
                c = allowed.get(str(name).strip().lower())
                try:
                    w = float(w)
                except (TypeError, ValueError):
                    continue
                if c and w > 0:
                    clean[c] = clean.get(c, 0) + w
            if clean:
                z = sum(clean.values())
                out[t] = {c: w / z for c, w in clean.items()}
        print(f"  neighbours: {len(out)}/{len(targets)}", flush=True)
    return out


def main():
    from .text_embed import embed_names
    p = argparse.ArgumentParser()
    p.add_argument("--zoo", default="zoo")
    p.add_argument("--meta", help="split JSON (names, seen, holdout) instead of <zoo>/meta.json")
    p.add_argument("--descriptions")
    p.add_argument("--out", default="data/neighbors.json")
    p.add_argument("--val", type=int, default=100, help="known classes treated as new, to tune the blend weight")
    p.add_argument("--k", type=int, default=20)
    a = p.parse_args()
    with open(a.meta or os.path.join(a.zoo, "meta.json")) as f:
        meta = json.load(f)
    names, seen = meta["names"], meta["seen"]
    meta.setdefault("holdout", sorted(meta.get("holdout_random", []) + meta.get("holdout_hard", [])))
    descs = json.load(open(a.descriptions)) if a.descriptions else None
    E = embed_names(names, ensemble=True, descriptions=descs)
    seen_names = [names[c] for c in seen]
    E_seen = E[seen]
    cache = json.load(open(a.out)) if os.path.exists(a.out) else {}
    # held-out classes: candidates from all known classes
    hold = [names[c] for c in meta["holdout"]]
    cand = dict(zip(hold, shortlist(E[meta["holdout"]], E_seen, seen_names, a.k)))
    # validation: known classes treated as new, candidates from the *other* known classes
    g = torch.Generator().manual_seed(0)
    val_ids = [seen[i] for i in torch.randperm(len(seen), generator=g)[:a.val].tolist()]
    val = [names[c] for c in val_ids]
    cand.update(dict(zip(val, shortlist(E[val_ids], E_seen, seen_names, a.k, exclude=val))))
    out = {"holdout": ask(hold, cand, cache=cache.get("holdout")), "val": ask(val, cand, cache=cache.get("val")),
           "val_classes": val}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"held-out answered {len(out['holdout'])}/{len(hold)}, validation {len(out['val'])}/{len(val)} -> {a.out}")


if __name__ == "__main__":
    main()
