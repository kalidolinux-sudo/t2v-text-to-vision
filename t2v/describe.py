"""Visual descriptions of classes written by an LLM on AMD TokenFactory (CuPL, Pratt et al., 2023).

A class name alone ("apple") gives CLIP little to work with. Averaging the CLIP embeddings of several
short visual descriptions ("a round red fruit with a short stem", ...) gives a much richer condition,
which matters most for classes the generator never saw.

    python -m t2v.describe --names_from zoo/meta.json --out data/descriptions.json
"""
import argparse
import json
import os
import re

import requests

SYSTEM = (
    "You write visual descriptions of object classes for an image classifier. For each class, give {n} "
    "short, different descriptions (8-15 words each) of what it looks like in a photo: shape, colour, "
    "texture, parts, typical setting. Mention features that tell it apart from similar classes. "
    "Reply with JSON only: {{\"<class>\": [\"...\", ...], ...}} using the class names exactly as given."
)


def _chat(messages, timeout=120):
    key = os.environ.get("TOKENFACTORY_API_KEY")
    base = os.environ.get("TOKENFACTORY_BASE_URL")
    model = os.environ.get("TOKENFACTORY_MODEL")
    if not (key and base and model):
        return None
    r = requests.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0.3, "max_tokens": 4000,
                            "response_format": {"type": "json_object"}, "messages": messages}, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def describe(names, n=5, batch=20, cache=None):
    """{name: [descriptions]} for every name; uses and extends `cache` (a dict) when given.
    Returns {} for names it could not describe (no endpoint configured, or a failed call)."""
    out = dict(cache or {})
    todo = [x for x in names if x not in out]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            text = _chat([{"role": "system", "content": SYSTEM.format(n=n)},
                          {"role": "user", "content": "Classes: " + ", ".join(c.replace("_", " ") for c in chunk)}])
        except requests.RequestException as e:
            print(f"describe: request failed ({e}); falling back to class names for {chunk}")
            continue
        if text is None:
            break
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0) if m else text)
        except json.JSONDecodeError:
            print(f"describe: could not parse the reply for {chunk}")
            continue
        lookup = {k.strip().lower().replace("_", " "): v for k, v in data.items()}
        for c in chunk:
            descs = lookup.get(c.replace("_", " ").lower())
            if isinstance(descs, list) and descs:
                out[c] = [str(d).strip() for d in descs if str(d).strip()][:n]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names_from", default="zoo/meta.json", help="zoo meta.json (uses its class names)")
    p.add_argument("--out", default="data/descriptions.json")
    p.add_argument("--n", type=int, default=5)
    a = p.parse_args()
    with open(a.names_from) as f:
        names = json.load(f)["names"]
    cache = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out = describe(names, a.n, cache=cache)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"{len(out)}/{len(names)} classes described -> {a.out}")


if __name__ == "__main__":
    main()
