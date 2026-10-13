"""Train the text-to-weights generator on a model zoo built by `t2v.zoo`."""
import argparse
import glob
import json
import os
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
    p.add_argument("--norm", default="tensor", choices=["tensor", "element"], help="body weight normalisation")
    p.add_argument("--pred", default="x0", choices=["x0", "eps"], help="denoiser target (x0 = Tina/G.pt signal prediction)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--tokens_per_step", type=int, default=8192, help="tokens denoised per step (memory knob)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--no_perm_aug", action="store_true")
    p.add_argument("--prompt_ensemble", action="store_true", help="average CLIP embeddings over several templates")
    p.add_argument("--emb_noise", type=float, default=0.0, help="std of Gaussian noise added to class embeddings")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (recommended on MI300X)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2000)
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
    class_emb = embed_names(meta["names"], dev, ensemble=a.prompt_ensemble)
    clip = max(tok.encode(flat[i:i + 256].to(dev)).abs().max().item() for i in range(0, len(flat), 256))
    gen = T2VGenerator(tok.layer_ids, tok.intra_ids, tok.slots, a.token_size, class_emb.shape[1], a.d_model,
                       a.backbone, a.backbone_layers, a.cond_layers,
                       channels=[int(c) for c in a.channels.split(",")], denoiser=a.denoiser,
                       mlp_hidden=a.mlp_hidden, mlp_depth=a.mlp_depth, pred=a.pred, clip=round(clip, 3)).to(dev)
    n_gen = sum(p.numel() for p in gen.parameters())
    print(f"tokens per model: {tok.num_tokens} x {a.token_size}; generator: {n_gen / 1e6:.1f}M params")

    opt = torch.optim.AdamW(gen.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    flat = flat.pin_memory() if dev.startswith("cuda") else flat
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def save(step):
        torch.save({"generator": gen.state_dict(), "config": gen.config, "tokenizer": tok.state(),
                    "arch": meta["arch"], "names": meta["names"], "prompt_ensemble": a.prompt_ensemble,
                    "step": step}, a.out)

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
        with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                            enabled=a.bf16):
            loss = gen.loss(x0, tok.valid, emb, mask, row_valid, a.tokens_per_step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.log_every == 0:
            print(f"step {step}  loss {run:.4f}  {step / (time.time() - t0):.1f} it/s", flush=True)
        if step % a.save_every == 0:
            save(step)
    save(a.steps)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
