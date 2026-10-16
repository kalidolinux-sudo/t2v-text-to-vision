"""TokenFactory agent step: a visual questionnaire for every class.

For each class the agent scores a fixed list of visual attributes (colour, texture, shape, parts, scene,
kind) from 0 (never) to 3 (always), as they appear in a small photo. The attribute vector is a second,
structured condition next to the CLIP text embedding: the row map (t2v.rowmap) learns from photos of the
seen classes how to turn it into classifier rows, so for a never-seen class the agent's answers become
its classifier directly (attribute-based zero-shot learning, Lampert et al., 2009).

    python -m t2v.attributes --out data/attributes_imagenet.json
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from . import tokenfactory

ATTRIBUTES = [
    # colour
    "black", "white", "gray", "brown", "red or orange", "yellow", "green", "blue", "pink or purple",
    "striped, spotted or patterned",
    # texture / material
    "furry or hairy", "feathered", "scaly or slimy skin", "shiny metal", "wood", "fabric or soft material",
    "glass, transparent or reflective", "plastic or painted surface",
    # shape / parts
    "round", "long and thin", "flat or rectangular", "boxy", "has legs", "four legs", "two legs", "wings",
    "fins or tail fin", "a long tail", "wheels", "a visible face or eyes", "handles, buttons or knobs",
    # size
    "fits in a hand", "about the size of a person", "larger than a car",
    # scene
    "in or under water", "in the sky", "on grass or in a forest", "on snow or ice", "on sand or in a desert",
    "on a road or street", "inside a home", "in a shop, kitchen or workshop", "at a shore or beach",
    "on rocks or mountains", "on a table or plate", "held or worn by a person", "with people nearby",
    # kind
    "a mammal", "a bird", "a fish or sea creature", "a reptile or amphibian", "an insect or spider",
    "a plant, flower or fungus", "food", "a vehicle", "a tool, device or machine", "a musical instrument",
    "clothing or an accessory", "furniture", "a building or structure",
]

SYSTEM = (
    "You are a computer-vision annotator. For each class, go through this fixed list of visual attributes and "
    "score how often each one is visible in a typical small photo of the class, including its usual "
    "surroundings: 1 = sometimes, 2 = often, 3 = always. Leave out attributes that never apply.\n"
    "Attributes:\n{attrs}\n"
    "Reply with JSON only: {{\"<class>\": {{\"<attribute>\": <1-3>, ...}}, ...}} with the class names exactly as "
    "given (without the hint in brackets) and the attribute names copied exactly from the list."
)


def _parse(text, chunk, n):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, TypeError):
        return {}
    lookup = {str(k).split(" (")[0].strip().lower(): v for k, v in data.items()}
    index = {a.lower(): i for i, a in enumerate(ATTRIBUTES)}
    out = {}
    for c in chunk:
        v = data.get(c, lookup.get(c.lower()))
        if not isinstance(v, dict):
            continue
        vec = [0] * n
        for a, x in v.items():
            i = index.get(str(a).strip().lower())
            try:
                if i is not None:
                    vec[i] = min(3, max(0, int(round(float(x)))))
            except (TypeError, ValueError):
                pass
        if any(vec):
            out[c] = vec
    return out


def annotate(names, hints=None, batch=10, workers=4, cache=None, retries=2, timeout=180):
    """{name: [len(ATTRIBUTES) scores 0-3, 0 = never]} for every name the agent answered. hints: {name: one
    description}, to tell apart homonyms ("crane" the bird vs the machine)."""
    out = dict(cache or {})
    system = SYSTEM.format(attrs="\n".join(f"- {a}" for a in ATTRIBUTES))
    todo = [c for c in names if c not in out]
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]

    def run(chunk):
        lines = "\n".join(f"- {c}" + (f" ({hints[c]})" if hints and hints.get(c) else "") for c in chunk)
        got = {}
        for _ in range(retries + 1):
            try:
                text = tokenfactory.chat([{"role": "system", "content": system},
                                          {"role": "user", "content": "Classes:\n" + lines}],
                                         temperature=0, max_tokens=6000, timeout=timeout)
            except Exception as e:  # a failed call is retried, then skipped
                print(f"attributes: request failed ({e})", flush=True)
                continue
            got.update(_parse(text, [c for c in chunk if c not in got], len(ATTRIBUTES)))
            if len(got) == len(chunk):
                break
        return got

    with ThreadPoolExecutor(workers) as ex:
        for i, got in enumerate(ex.map(run, chunks)):
            out.update(got)
            if i % 10 == 0:
                print(f"  attributes: {len(out)}/{len(names)}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/imagenet_split.json")
    p.add_argument("--descriptions", default="data/descriptions_imagenet.json")
    p.add_argument("--out", default="data/attributes_imagenet.json")
    p.add_argument("--workers", type=int, default=8, help="parallel TokenFactory requests")
    p.add_argument("--batch", type=int, default=10, help="classes per request")
    a = p.parse_args()
    if tokenfactory.config() is None:
        raise SystemExit("Set TOKENFACTORY_API_KEY first (see t2v/tokenfactory.py).")
    names = json.load(open(a.split))["names"]
    descs = json.load(open(a.descriptions)) if a.descriptions else {}
    hints = {c: (descs.get(c) or [None])[0] for c in names}
    cache = json.load(open(a.out))["scores"] if os.path.exists(a.out) else None
    scores = dict(cache or {})
    for i in range(0, len(names), 200):  # saved every 200 classes, so an interrupted run resumes
        scores = annotate(names[i:i + 200], hints, batch=a.batch, workers=a.workers, cache=scores)
        json.dump({"attributes": ATTRIBUTES, "model": tokenfactory.model_name(), "scores": scores},
                  open(a.out, "w"), indent=0)
    print(f"{len(scores)}/{len(names)} classes annotated, {tokenfactory.STATS['calls']} TokenFactory calls, "
          f"{tokenfactory.STATS['seconds']:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
