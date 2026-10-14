"""TokenFactory role 3: visual analogies for classes the generator has never seen.

The diagnostics showed that for a new class the generator snaps its classifier row to some known class.
Here an LLM on AMD TokenFactory reasons about it explicitly: from a CLIP shortlist of known classes it
picks the few that look most alike in a photo, with weights ("zebra: 0.6 horse, 0.4 tiger"). The row
prior for the new class is then the weighted mix of those classes' learned rows, blended with the ridge
prior by a weight chosen with cross-validation on known classes (fit_blend, called by train_gen).

    python -m t2v.neighbors --zoo zoo --descriptions data/descriptions.json --out data/neighbors.json
"""
import argparse
import json
import os
import re

import requests
import torch
import torch.nn.functional as F

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


def class_row_table(rows, classes, num_classes):
    """Mean learned classifier row of every class over the zoo. rows (N, K, F+1), classes (N, K) with -1
    padding -> table (num_classes, F+1); classes never trained on stay zero."""
    valid = classes >= 0
    r, c = rows[valid].float(), classes[valid]
    table = torch.zeros(num_classes, r.shape[-1]).index_add_(0, c, r)
    count = torch.zeros(num_classes).index_add_(0, c, torch.ones(len(c)))
    return table / count.clamp_min(1)[:, None]


def mix_rows(picks, table, name2id):
    """{candidate name: weight} -> the weighted sum of those classes' rows, or None."""
    ids = [(name2id[n], w) for n, w in (picks or {}).items() if n in name2id]
    if not ids:
        return None
    return sum(w * table[i] for i, w in ids) / sum(w for _, w in ids)


def clip_mix(target_emb, cand_emb, cand_ids, table, k=3, tau=0.03, exclude=None):
    """Baseline without the LLM: softmax(similarity / tau) over the k most CLIP-similar candidates."""
    dev = target_emb.device
    sims = target_emb @ cand_emb.T
    ids = torch.as_tensor(cand_ids, device=dev)
    if exclude is not None:
        sims[ids[None, :] == torch.as_tensor(exclude, device=dev)[:, None]] = -2.0
    top = sims.topk(k, dim=1)
    w = (top.values / tau).softmax(1)
    return torch.einsum("nk,nkf->nf", w, table.to(dev)[ids[top.indices]])


def fit_blend(class_emb, rows, cls_ids, table, lam, val_ids, val_picks, names, seen_ids,
              betas=tuple(i / 10 for i in range(11)), taus=(0.01, 0.03, 0.1), k=3):
    """Choose how much of the neighbour row to mix into the ridge prior, on known classes treated as new.
    The ridge map is refitted without those classes, so each is predicted as if it had never been seen;
    the score is the cosine to its real (zoo-mean) row. Returns the chosen weights and all scores, for the
    TokenFactory neighbours and for a CLIP nearest-neighbour baseline that uses no LLM.
    class_emb (C, E), rows (M, F+1) and cls_ids (M,): every zoo row and its class."""
    name2id = {names[i]: i for i in seen_ids}  # classes with a learned row
    V = [c for c in val_ids if c in name2id.values() and mix_rows(val_picks.get(names[c]), table, name2id) is not None]
    if len(V) < 10:
        return None
    Vt = torch.tensor(V)
    keep = ~torch.isin(cls_ids, Vt)
    X, Y = class_emb[cls_ids[keep]], rows[keep].float()
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ Y)
    ridge, truth = class_emb[Vt] @ W, table[Vt]

    def score(P):
        cos = F.cosine_similarity(P, truth, dim=-1).mean().item()
        ret = (F.normalize(P, dim=-1) @ F.normalize(truth, dim=-1).T).argmax(1).eq(torch.arange(len(V))).float().mean().item()
        return {"row_cosine": round(cos, 4), "row_retrieval": round(ret, 4)}

    def best(nb):
        s = {b: score((1 - b) * ridge + b * nb) for b in betas}
        b = max(s, key=lambda x: s[x]["row_cosine"])
        return b, s[b]

    llm = torch.stack([mix_rows(val_picks[names[c]], table, name2id) for c in V])
    b_llm, s_llm = best(llm)
    clip_best = None
    for tau in taus:
        nb = clip_mix(class_emb[Vt], class_emb[seen_ids], seen_ids, table, k, tau, exclude=V)
        b, s = best(nb)
        if clip_best is None or s["row_cosine"] > clip_best[2]["row_cosine"]:
            clip_best = (tau, b, s)
    return {"llm_beta": b_llm, "clip_beta": clip_best[1], "clip_tau": clip_best[0], "clip_k": k,
            "auto": "llm" if s_llm["row_cosine"] >= clip_best[2]["row_cosine"] else "clip",
            "cv": {"validation_classes": len(V), "ridge_only": score(ridge), "llm_neighbours_only": score(llm),
                   f"ridge+llm (beta {b_llm})": s_llm,
                   f"ridge+clip_knn (beta {clip_best[1]}, tau {clip_best[0]})": clip_best[2]}}


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
