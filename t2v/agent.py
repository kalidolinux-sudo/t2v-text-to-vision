"""Watch the AMD TokenFactory agent work, live, in one command (needs TOKENFACTORY_API_KEY):

    python -m t2v.agent "I need a model that tells zebras, grand pianos and killer whales apart"

It runs the four agent steps that feed the generator and prints what TokenFactory returned:
    1 understand  free-form request (any language) -> English class names
    2 ground      names mapped onto the generator's vocabulary when they mean the same category
    3 describe    visual descriptions of every class (what the generator is conditioned on)
    4 analogise   for classes outside the 900 training classes, the known classes they look like
No weights are generated here; the app, the notebook and `python -m t2v.generate` do that.
"""
import argparse
import json
import time

from . import neighbors as N
from . import tokenfactory
from .describe import describe
from .ground import ground
from .llm import parse_classes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request")
    p.add_argument("--split", default="data/imagenet_split.json", help="ImageNet class split (seen = training classes)")
    p.add_argument("--descriptions", default="data/descriptions_imagenet.json", help="stored TokenFactory descriptions")
    a = p.parse_args()
    if tokenfactory.config() is None:
        raise SystemExit("Set TOKENFACTORY_API_KEY first (see t2v/tokenfactory.py).")
    t0 = time.time()
    print(f"agent: {tokenfactory.model_name()} on AMD TokenFactory\n")

    classes, info = parse_classes(a.request)
    print(f"1 understand  {a.request!r}\n  -> {classes}" + (f"   ({info['task']})" if info.get("task") else ""))

    split = json.load(open(a.split))
    mapping = ground(classes, split["names"])
    moved = [(c, mapping[c]) for c in classes if mapping.get(c) and mapping[c] != c]
    print("\n2 ground       " + (", ".join(f"{x} -> {y}" for x, y in moved) if moved else "nothing to map: names kept as given"))
    classes = list(dict.fromkeys(mapping.get(c) or c for c in classes))

    descs = {c: d for c, d in (info.get("descriptions") or {}).items() if d}
    stored = json.load(open(a.descriptions)) if a.descriptions else {}
    descs.update({c: stored[c] for c in classes if c in stored and not descs.get(c)})  # written by TokenFactory at build time
    missing = [c for c in classes if not descs.get(c)]
    if missing:
        descs.update(describe(missing))
    print("\n3 describe")
    for c in classes:
        for d in (descs.get(c) or ["(no description returned)"])[:2]:
            print(f"  {c}: {d}")

    seen = [split["names"][i] for i in split["seen"]]
    canon = {n.lower(): n for n in seen}
    new = [c for c in classes if c.lower() not in canon]
    print("\n4 analogise")
    if not new:
        print("  every class is a training class: no analogies needed")
    else:
        from .text_embed import embed_names  # CLIP proposes candidates, the agent picks the look-alikes
        cands = N.shortlist(embed_names(new), embed_names(seen), seen, 20)  # names vs names
        picks = N.ask(new, dict(zip(new, cands)))
        for c in new:
            look = ", ".join(f"{n} {w:.0%}" for n, w in (picks.get(c) or {}).items()) or "(no valid answer)"
            print(f"  {c} looks like: {look}")

    s = tokenfactory.STATS
    print(f"\n{s['calls']} TokenFactory calls, {s['seconds']:.1f} s of {time.time() - t0:.1f} s total")


if __name__ == "__main__":
    main()
