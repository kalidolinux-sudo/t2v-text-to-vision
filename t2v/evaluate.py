"""Accuracy of generated models on held-out tasks, against Tina's baselines.

  generated (mean / best of n)  weights straight from the generator, no training, no images
  oracle                        a p-model actually fine-tuned on the task (upper reference)
  selection                     the generic model's head restricted to the task's classes
                                (impossible when a class was never seen, like TAPER in Tina)
"""
import argparse
import json
import os
import time

import torch

from . import target
from .data import Cifar100, prep
from .generate import generate_flat, load_generator
from .text_embed import embed_names
from .train_gen import load_split
from .zoo import accuracy, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/generator.pt")
    p.add_argument("--zoo", default="zoo")
    p.add_argument("--data", default="data")
    p.add_argument("--splits", default="eval_seen,eval_unseen,eval_novel")
    p.add_argument("--max_tasks", type=int, default=100)
    p.add_argument("--samples", type=int, default=5, help="generated models per task")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--fp16", action="store_true", help="fp16 sampling on GPU")
    p.add_argument("--out", default="runs/eval.json")
    a = p.parse_args()

    dev = pick_device(a.device)
    gen, tok, ck = load_generator(a.ckpt, dev)
    arch = ck["arch"]
    data = Cifar100(a.data, dev, features=target.uses_features(arch))
    generic = target.build(arch, 100).to(dev)
    generic.load_state_dict(torch.load(os.path.join(a.zoo, "generic.pt"), map_location=dev)["state"])
    with open(os.path.join(a.zoo, "meta.json")) as f:
        seen = set(json.load(f)["seen"])
    class_emb = embed_names(ck["names"], dev, ensemble=ck.get("prompt_ensemble", False),
                            descriptions=ck.get("descriptions"))
    net = target.build(arch).to(dev)
    zs_emb = embed_names(ck["names"], dev, ensemble=True)  # CLIP zero-shot uses the prompt ensemble

    report = {}
    for split in a.splits.split(","):
        try:
            flat, classes, oracle = load_split(a.zoo, split)
        except SystemExit as e:
            print(e)
            continue
        rows, gen_time = [], 0.0
        for i in range(min(a.max_tasks, len(flat))):
            cls = [c for c in classes[i].tolist() if c >= 0]
            xt, yt = data.subset(cls, train=False)
            t0 = time.time()
            flats = generate_flat(gen, tok, cls, n=a.samples, steps=a.steps, emb=class_emb[cls],
                                  prior_emb=zs_emb[cls], arch=arch, amp=a.fp16)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            gen_time += time.time() - t0
            accs = [accuracy(target.load_flat(net, f), xt, yt, k=len(cls)) for f in flats]
            sel = None
            if all(c in seen for c in cls):
                with torch.no_grad():
                    logits = generic(prep(xt))[:, cls]
                sel = (logits.argmax(1) == yt).float().mean().item()
            zs = None
            if target.uses_features(arch):
                zs = ((xt @ zs_emb[cls].T).argmax(1) == yt).float().mean().item()
            rows.append({"classes": cls, "generated": accs, "oracle": oracle[i].item(), "selection": sel,
                         "clip_zero_shot": zs})
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        summary = {
            "tasks": len(rows),
            "generated_mean": mean([mean(r["generated"]) for r in rows]),
            "generated_best": mean([max(r["generated"]) for r in rows]),
            "oracle": mean([r["oracle"] for r in rows]),
            "selection": mean([r["selection"] for r in rows if r["selection"] is not None]),
            "clip_zero_shot": mean([r["clip_zero_shot"] for r in rows if r["clip_zero_shot"] is not None]),
            "chance": mean([1 / len(r["classes"]) for r in rows]),
            "seconds_per_model": gen_time / max(1, len(rows) * a.samples),
        }
        report[split] = {"summary": summary, "tasks": rows}
        print(f"[{split}] " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                         for k, v in summary.items()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"arch": arch, "num_params": tok.num_params, **report}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
