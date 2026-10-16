"""TokenFactory agent step 2: ground the user's words in the generator's vocabulary.

People do not type ImageNet class names: they write "orca", "sneakers", "tabby cat", "مطرقة". A name the
generator was trained on gets its learned classifier row (the known-class gate); any other name only gets
a row estimated from its text. Here the agent maps each requested name onto the vocabulary when it
denotes the same visual category (a synonym, a more common name, a translation) and keeps it as a new
class when it is a different or broader category ("dog" is not one breed). CLIP proposes the candidates;
the agent decides. One TokenFactory call per request.
"""
import json
import re

import torch

from . import tokenfactory

SYSTEM = (
    "You map class names from a user's request for an image classifier onto a fixed vocabulary of image "
    "classes. For each name you get vocabulary candidates. If the name denotes the same visual category as one "
    "candidate (a synonym, a more common or more technical name, or a translation of it), answer that candidate "
    "copied exactly. If it is a different category, or a broader one that covers several candidates (\"dog\", "
    "\"bird\"), answer null. Reply with JSON only: {\"<name>\": \"<candidate>\" or null, ...}"
)

_vocab_emb = {}


@torch.no_grad()
def vocab_embeddings(vocab, device="cpu"):
    key = (len(vocab), device)
    if key not in _vocab_emb:
        from .text_embed import embed_names
        _vocab_emb[key] = embed_names(vocab, device)
    return _vocab_emb[key]


def candidates(names, vocab, k=8, device="cpu"):
    from .text_embed import embed_names
    emb = embed_names(names, device)
    sims = emb @ vocab_embeddings(vocab, device).T
    return {n: [vocab[j] for j in sims[i].argsort(descending=True)[:k].tolist()] for i, n in enumerate(names)}


def ground(names, vocab, k=8, device="cpu", timeout=120, batch=20):
    """{name: vocabulary name or None} for every name not already in the vocabulary (case-insensitive).
    Names already in the vocabulary map to themselves; without a key nothing else is mapped."""
    canon = {v.lower(): v for v in vocab}
    out = {n: canon[n.lower()] for n in names if n.lower() in canon}
    todo = [n for n in names if n.lower() not in canon]
    if not todo or tokenfactory.config() is None:
        return {**out, **{n: None for n in todo}}
    cands = candidates(todo, vocab, k, device)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        prompt = "\n".join(f"Name: {n}\nCandidates: {', '.join(cands[n])}" for n in chunk)
        try:
            text = tokenfactory.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                     temperature=0, max_tokens=1500, timeout=timeout)
            m = re.search(r"\{.*\}", text or "", re.S)
            data = json.loads(m.group(0)) if m else {}
        except Exception as e:  # grounding is an improvement, never a blocker
            print(f"ground: request failed ({e})")
            data = {}
        lookup = {str(key).strip().lower(): v for key, v in data.items()}
        for n in chunk:
            v = lookup.get(n.lower())
            allowed = {c.lower(): c for c in cands[n]}
            out[n] = allowed.get(str(v).strip().lower()) if isinstance(v, str) else None
    return out
