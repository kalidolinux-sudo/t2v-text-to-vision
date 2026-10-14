"""Where do the parameters and the generation time of a T2V generator go?

    python -m t2v.profile --ckpt ckpt/generator.pt            # a trained generator
    python -m t2v.profile --arch resnet20 --joint_layers 4    # a fresh one with the given layout
"""
import argparse
import time
from collections import defaultdict

import torch

from . import target
from .generator import T2VGenerator
from .generate import load_generator
from .tokenizer import ParamTokenizer


def count(module):
    return sum(p.numel() for p in module.parameters())


def build_fresh(a, dev):
    spec = target.param_spec(target.build(a.arch))
    flat = torch.cat([target.flatten(target.build(a.arch))[None] for _ in range(2)])
    tok = ParamTokenizer(spec, a.token_size, ParamTokenizer.compute_stats(flat, spec), target.MAX_CLASSES).to(dev)
    gen = T2VGenerator(tok.layer_ids, tok.intra_ids, tok.slots, a.token_size, 512, a.d_model, a.backbone,
                       denoiser=a.denoiser, mlp_hidden=a.mlp_hidden, pred="x0", clip=5.0,
                       joint_layers=a.joint_layers, valid=tok.valid).to(dev)
    return gen.eval(), tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--arch", default="resnet20")
    p.add_argument("--token_size", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--backbone", default="lstm")
    p.add_argument("--denoiser", default="mlp")
    p.add_argument("--mlp_hidden", type=int, default=1024)
    p.add_argument("--joint_layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = a.device
    gen, tok = (load_generator(a.ckpt, dev)[:2] if a.ckpt else build_fresh(a, dev))
    print(f"tokens per model: {tok.num_tokens} x {tok.token_size}  ({tok.num_params:,} generated params)\n")

    parts = {"class encoder": gen.cond, "position + prototype layers": [gen.pos_proj, gen.xattn, gen.xnorm, gen.proto_out],
             "recurrent backbone": gen.backbone, "per-token denoiser": gen.denoiser}
    if gen.config.get("joint_layers", 0):
        parts["joint transformer"] = [gen.joint_in, gen.joint_time, gen.joint, gen.joint_out]
    total = count(gen)
    print(f"{'component':30s} {'params':>12s} {'share':>7s}")
    for name, mods in parts.items():
        n = sum(count(m) for m in (mods if isinstance(mods, list) else [mods]))
        print(f"{name:30s} {n:12,d} {100 * n / total:6.1f}%")
    print(f"{'total':30s} {total:12,d}\n")

    # time each stage of one generation (one model, `steps` DDIM steps)
    timings = defaultdict(float)

    def sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    def timed(name, fn, *args):
        sync()
        t0 = time.perf_counter()
        out = fn(*args)
        sync()
        timings[name] += time.perf_counter() - t0
        return out

    K = tok.K
    emb = torch.nn.functional.normalize(torch.randn(1, K, gen.config["clip_dim"], device=dev), dim=-1)
    mask = torch.ones(1, K, dtype=torch.bool, device=dev)
    joint = gen.config.get("joint_layers", 0) > 0
    with torch.no_grad(), torch.autocast("cuda" if dev.startswith("cuda") else "cpu", dtype=torch.float16,
                                         enabled=a.fp16 and dev.startswith("cuda")):
        for rep in range(3):  # first pass warms up kernels; report the last two
            if rep == 1:
                timings.clear()
            proto, classes = timed("prototypes (once)", gen.prototypes, emb, mask, True)
            T, d = proto.shape[1], proto.shape[2]
            x = torch.randn(T, gen.config["token_size"], device=dev)
            for t in torch.linspace(len(gen.alpha_bar) - 1, 0, a.steps, device=dev).round().long():
                p_ = proto.reshape(T, d)
                if joint:
                    p_ = timed("joint transformer (x steps)", gen.context, x.view(1, T, -1), t.expand(1), proto,
                               classes).reshape(T, d)
                x0 = timed("per-token denoiser (x steps)", gen.denoiser, x, t.expand(T), p_).float()
                ab = gen.alpha_bar[t]
                x = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x)
    tot = sum(timings.values()) / 2
    print(f"{'generation stage':30s} {'ms':>9s} {'share':>7s}   ({a.steps} steps, {'fp16' if a.fp16 else 'fp32'}, {dev})")
    for name, v in timings.items():
        print(f"{name:30s} {1000 * v / 2:9.1f} {100 * v / 2 / tot:6.1f}%")
    print(f"{'total':30s} {1000 * tot:9.1f}")


if __name__ == "__main__":
    main()
