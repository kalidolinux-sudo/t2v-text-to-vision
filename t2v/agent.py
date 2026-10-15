"""Watch the AMD TokenFactory agent work, live, in one command (needs TOKENFACTORY_API_KEY):

    python -m t2v.agent "I need a model that tells zebras, grand pianos and killer whales apart"

It runs the three agent steps that feed the generator and prints what TokenFactory returned:
    1 understand  free-form request (any language) -> English class names
    2 describe    visual descriptions of every class (what the generator is conditioned on)
    3 analogise   for classes outside the 900 training classes, the known classes they look like
No weights are generated here; the app, the notebook and `python -m t2v.generate` do that.
"""
import argparse
import json
import time

from . import neighbors as N
from . import tokenfactory
from .describe import describe
from .llm import parse_classes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request")
    p.add_argument("--split", default="data/imagenet_split.json", help="ImageNet class split (seen = training classes)")
    a = p.parse_args()
    if tokenfactory.config() is None:
        raise SystemExit("Set TOKENFACTORY_API_KEY first (see t2v/tokenfactory.py).")
    t0 = time.time()
    print(f"agent: {tokenfactory.model_name()} on AMD TokenFactory\n")

    classes, info = parse_classes(a.request)
    print(f"1 understand  {a.request!r}\n  -> {classes}" + (f"   ({info['task']})" if info.get("task") else ""))

    descs = {c: d for c, d in (info.get("descriptions") or {}).items() if d}
    missing = [c for c in classes if not descs.get(c)]
    if missing:
        descs.update(describe(missing))
    print("\n2 describe")
    for c in classes:
        for d in (descs.get(c) or ["(no description returned)"])[:2]:
            print(f"  {c}: {d}")

    split = json.load(open(a.split))
    seen = [split["names"][i] for i in split["seen"]]
    canon = {n.lower(): n for n in seen}
    new = [c for c in classes if c.lower() not in canon]
    print("\n3 analogise")
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
