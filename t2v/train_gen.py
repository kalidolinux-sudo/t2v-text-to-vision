"""Train the text-to-weights generator on a model zoo built by `t2v.zoo`."""
import argparse
import glob
import json
import math
import os
import signal
import time

import torch

from .generator import T2VGenerator
from .text_embed import embed_names
from .tokenizer import ParamTokenizer
from .zoo import pick_device
from . import target


def load_split(zoo, split):
    flats, classes, accs = [], [], []
    for path in sorted(glob.glob(os.path.join(zoo, f"{split}_shard*.pt"))):
        s = torch.load(path)
        flats += s["flat"]
        classes += s["classes"]
        accs += s["acc"]
    if not flats:
        raise SystemExit(f"no {split} shards in {zoo}")
    return torch.stack(flats).float(), torch.tensor(classes), torch.tensor(accs)


def class_batch(class_emb, classes):
    """classes (B,K) with -1 padding -> (emb (B,K,E), mask (B,K), k (B,))."""
    mask = classes >= 0
    return class_emb[classes.clamp_min(0)], mask, mask.sum(1)


def permute_classes(x0, classes, k, head_pos):
    """Tina's classifier augmentation: shuffle class order and the matching head rows together."""
    B, K = classes.shape
    keys = torch.rand(B, K, device=x0.device)
    keys[torch.arange(K, device=x0.device)[None] >= k[:, None]] = 2.0  # padding stays at the end
    perm = keys.argsort(1)
    rows = x0[torch.arange(B, device=x0.device)[:, None], head_pos[perm]]
    x0 = x0.clone()
    x0[:, head_pos] = rows
    return x0, classes.gather(1, perm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zoo", default="zoo")
    p.add_argument("--out", default="ckpt/generator.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--token_size", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--backbone", default="lstm", choices=["lstm", "gru", "transformer", "mamba"])
    p.add_argument("--backbone_layers", type=int, default=2)
    p.add_argument("--cond_layers", type=int, default=2)
    p.add_argument("--channels", default="64,128,256", help="conv denoiser widths")
    p.add_argument("--denoiser", default="conv", choices=["conv", "mlp"])
    p.add_argument("--mlp_hidden", type=int, default=512)
    p.add_argument("--mlp_depth", type=int, default=3)
    p.add_argument("--norm", default="tensor", choices=["tensor", "element", "zscore"],
                   help="body weight normalisation (zscore recommended for BatchNorm nets like ResNet)")
    p.add_argument("--pred", default="x0", choices=["x0", "eps"], help="denoiser target (x0 = Tina/G.pt signal prediction)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--tokens_per_step", type=int, default=8192, help="tokens denoised per step (memory knob)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--cosine", action="store_true", help="cosine-decay the LR to 0 after warmup")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay for the saved weights (e.g. 0.999); 0 = off")
    p.add_argument("--no_perm_aug", action="store_true")
    p.add_argument("--prompt_ensemble", action="store_true", help="average CLIP embeddings over several templates")
    p.add_argument("--descriptions", help="JSON of LLM-written visual descriptions per class (t2v.describe)")
    p.add_argument("--emb_noise", type=float, default=0.0, help="std of Gaussian noise added to class embeddings")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (recommended on MI300X)")
    p.add_argument("--fp16", action="store_true", help="fp16 autocast + loss scaling (for T4/V100-class GPUs)")
    p.add_argument("--joint_layers", type=int, default=0,
                   help="hybrid mode: Tina/G.pt-style transformer layers over all noisy tokens (0 = pure RPG)")
    p.add_argument("--head_weight", type=float, default=1.0,
                   help="loss weight of classifier-row tokens (they are also always included in each step)")
    p.add_argument("--min_snr", type=float, default=0.0, help="Min-SNR-gamma loss weighting, e.g. 5; 0 = off")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000,
                   help="checkpoint period; also: kill -USR1 <pid> saves now, Ctrl-C / kill saves and stops")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    with open(os.path.join(a.zoo, "meta.json")) as f:
        meta = json.load(f)
    flat, classes, accs = load_split(a.zoo, "train")
    print(f"zoo: {len(flat)} p-models x {flat.shape[1]:,} params (mean acc {accs.mean():.3f})")

    spec = target.param_spec(target.build(meta["arch"]))
    tok = ParamTokenizer(spec, a.token_size, ParamTokenizer.compute_stats(flat, spec, a.norm), meta["max_classes"]).to(dev)
    descriptions = json.load(open(a.descriptions)) if a.descriptions else None
    class_emb = embed_names(meta["names"], dev, ensemble=a.prompt_ensemble, descriptions=descriptions)
    clip = max(tok.encode(flat[i:i + 256].to(dev)).abs().max().item() for i in range(0, len(flat), 256))
    gen = T2VGenerator(tok.layer_ids, tok.intra_ids, tok.slots, a.token_size, class_emb.shape[1], a.d_model,
                       a.backbone, a.backbone_layers, a.cond_layers,
                       channels=[int(c) for c in a.channels.split(",")], denoiser=a.denoiser,
                       mlp_hidden=a.mlp_hidden, mlp_depth=a.mlp_depth, pred=a.pred, clip=round(clip, 3),
                       joint_layers=a.joint_layers, valid=tok.valid).to(dev)
    n_gen = sum(p.numel() for p in gen.parameters())
    print(f"tokens per model: {tok.num_tokens} x {a.token_size}; generator: {n_gen / 1e6:.1f}M params")

    opt = torch.optim.AdamW(gen.parameters(), lr=a.lr, weight_decay=0.0)
    def lr_lambda(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        if a.cosine:
            return 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = None
    if a.ema > 0:
        ema = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    flat = flat.pin_memory() if dev.startswith("cuda") else flat
    use_fp16 = a.fp16 and dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def save(step):
        weights = ema if ema is not None else gen.state_dict()
        torch.save({"generator": weights, "config": gen.config, "tokenizer": tok.state(),
                    "arch": meta["arch"], "names": meta["names"], "prompt_ensemble": a.prompt_ensemble,
                    "descriptions": descriptions,
                    "step": step}, a.out)

    flags = {"stop": False, "snapshot": False}
    signal.signal(signal.SIGINT, lambda *_: flags.update(stop=True))
    signal.signal(signal.SIGTERM, lambda *_: flags.update(stop=True))
    signal.signal(signal.SIGUSR1, lambda *_: flags.update(snapshot=True))

    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(flat), (a.bs,))
        x0 = tok.encode(flat[idx].to(dev, non_blocking=True))
        cls = classes[idx].to(dev)
        k = (cls >= 0).sum(1)
        if not a.no_perm_aug:
            x0, cls = permute_classes(x0, cls, k, tok.head_pos)
        emb, mask, k = class_batch(class_emb, cls)
        if a.emb_noise > 0:
            emb = torch.nn.functional.normalize(emb + a.emb_noise * torch.randn_like(emb), dim=-1)
        row_valid = (tok.slots[None] < 0) | (tok.slots[None] < k[:, None])
        with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=amp_dtype,
                            enabled=a.bf16 or use_fp16):
            loss = gen.loss(x0, tok.valid, emb, mask, row_valid, a.tokens_per_step, a.min_snr, a.head_weight)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].lerp_(v, 1 - a.ema)
                    else:
                        ema[k].copy_(v)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.log_every == 0:
            print(f"step {step}  loss {run:.4f}  {step / (time.time() - t0):.1f} it/s", flush=True)
        if step % a.save_every == 0 or flags["snapshot"]:
            save(step)
            flags["snapshot"] = False
        if flags["stop"]:
            save(step)
            print(f"stopped at step {step}, saved {a.out}")
            return
    save(a.steps)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
