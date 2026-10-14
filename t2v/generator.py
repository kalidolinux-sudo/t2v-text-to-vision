"""Text-conditioned recurrent parameter diffusion: Tina's conditioning on RPG's scalable generator.

  class names --CLIP--> class tokens --set encoder--> condition
  token positions (layer, chunk) + condition + per-row class embedding
      --recurrent backbone (LSTM / GRU / Transformer / Mamba)--> one prototype per parameter token
  every token is then denoised independently by a small 1-D conv U-Net conditioned on its prototype,
  so memory grows with the number of tokens processed per step, not with the model size.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos(pos, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=pos.device) / half)
    ang = pos.float()[:, None] * freq[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class CondEncoder(nn.Module):
    """Set encoder over class embeddings. No positional encoding, so it is permutation invariant;
    missing classes (tasks with fewer than K classes) become a learned placeholder, as in Tina."""

    def __init__(self, clip_dim, d, layers, heads):
        super().__init__()
        self.inp = nn.Linear(clip_dim, d)
        self.placeholder = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, emb, cls_mask):
        h = torch.where(cls_mask[..., None], self.inp(emb), self.placeholder)
        return self.enc(h)


class Backbone(nn.Module):
    def __init__(self, kind, d, layers, heads):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.net = getattr(nn, kind.upper())(d, d, layers, batch_first=True)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.net = nn.TransformerEncoder(layer, layers)
        elif kind == "mamba":  # RPG's default; needs `pip install mamba-ssm` (has ROCm support)
            from mamba_ssm import Mamba
            self.net = nn.ModuleList([Mamba(d_model=d) for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        else:
            raise ValueError(kind)

    def forward(self, h):
        if self.kind in ("lstm", "gru"):
            return h + self.net(h)[0]
        if self.kind == "transformer":
            return self.net(h)
        for norm, layer in zip(self.norms, self.net):
            h = h + layer(norm(h))
        return h


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout, d):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 3, padding=1)
        self.film = nn.Linear(d, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(cond)[..., None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return h + self.skip(x)


class ConvDenoiser(nn.Module):
    """1-D U-Net over one parameter token, conditioned on (prototype + timestep) via FiLM."""

    def __init__(self, token_size, d, channels):
        super().__init__()
        assert token_size % 2 ** (len(channels) - 1) == 0
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.d = d
        self.inp = nn.Conv1d(1, channels[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        cin = channels[0]
        for i, c in enumerate(channels):
            self.down.append(FiLMBlock(cin, c, d))
            self.pool.append(nn.Conv1d(c, c, 3, stride=2, padding=1) if i < len(channels) - 1 else nn.Identity())
            cin = c
        self.mid = FiLMBlock(cin, cin, d)
        self.up = nn.ModuleList()
        for c in reversed(channels):
            self.up.append(FiLMBlock(cin + c, c, d))
            cin = c
        self.out = nn.Conv1d(cin, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, proto):
        cond = proto + self.time(sincos(t, self.d))
        h, skips = self.inp(x[:, None]), []
        for block, pool in zip(self.down, self.pool):
            h = block(h, cond)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, cond)
        for block in self.up:
            s = skips.pop()
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = block(torch.cat([h, s], 1), cond)
        return self.out(h)[:, 0]


class MLPDenoiser(nn.Module):
    """Residual MLP over one parameter token with FiLM conditioning; much cheaper than the U-Net
    for small tokens (and on CPU)."""

    def __init__(self, token_size, d, hidden=512, depth=3):
        super().__init__()
        self.d = d
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.inp = nn.Linear(token_size, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden, elementwise_affine=False) for _ in range(depth)])
        self.films = nn.ModuleList([nn.Linear(d, 2 * hidden) for _ in range(depth)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
                                   for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, token_size))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, proto):
        cond = F.silu(proto + self.time(sincos(t, self.d)))
        h = self.inp(x)
        for norm, film, mlp in zip(self.norms, self.films, self.mlps):
            scale, shift = film(cond).chunk(2, -1)
            h = h + mlp(norm(h) * (1 + scale) + shift)
        return self.out(h)


def cosine_alpha_bar(steps, s=0.008):
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    f = torch.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(0, 0.999)
    return torch.cumprod(1 - betas, 0).float()


class T2VGenerator(nn.Module):
    def __init__(self, layer_ids, intra_ids, slots, token_size, clip_dim=512, d_model=512, backbone="lstm",
                 backbone_layers=2, cond_layers=2, heads=8, channels=(64, 128, 256), diffusion_steps=1000,
                 denoiser="conv", mlp_hidden=512, mlp_depth=3, pred="x0", clip=None, joint_layers=0,
                 valid=None):
        super().__init__()
        self.config = dict(token_size=token_size, clip_dim=clip_dim, d_model=d_model, backbone=backbone,
                           backbone_layers=backbone_layers, cond_layers=cond_layers, heads=heads,
                           channels=tuple(channels), diffusion_steps=diffusion_steps, denoiser=denoiser,
                           mlp_hidden=mlp_hidden, mlp_depth=mlp_depth, pred=pred, clip=clip,
                           joint_layers=joint_layers)
        self.register_buffer("slots", slots.clone(), persistent=False)
        half = d_model // 2
        self.register_buffer("pos", torch.cat([sincos(layer_ids, half), sincos(intra_ids, half)], -1), persistent=False)
        self.register_buffer("alpha_bar", cosine_alpha_bar(diffusion_steps), persistent=False)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.cond = CondEncoder(clip_dim, d_model, cond_layers, heads)
        self.backbone = Backbone(backbone, d_model, backbone_layers, heads)
        self.xattn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.xnorm = nn.LayerNorm(d_model)
        self.proto_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(),
                                       nn.Linear(d_model, d_model))
        self.denoiser = (ConvDenoiser(token_size, d_model, channels) if denoiser == "conv"
                         else MLPDenoiser(token_size, d_model, mlp_hidden, mlp_depth))
        # Hybrid mode (joint_layers > 0): as in Tina / G.pt, every noisy parameter token of a model is
        # embedded and a transformer lets the tokens attend to each other (and to the class tokens) at each
        # denoising step, so the layers of one network are denoised consistently. Its output refines the
        # RPG prototypes, and the cheap per-token denoiser above still decodes each token.
        if joint_layers > 0:
            if valid is None:
                raise ValueError("joint mode needs the tokenizer's valid mask")
            self.register_buffer("valid", valid.clone(), persistent=False)
            self.joint_in = nn.Linear(token_size, d_model)
            self.joint_time = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, dropout=0.0, batch_first=True,
                                               norm_first=True)
            self.joint = nn.TransformerDecoder(layer, joint_layers)  # self-attention over tokens + cross-attention to classes
            self.joint_out = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.joint_out.weight)
            nn.init.zeros_(self.joint_out.bias)

    def prototypes(self, emb, cls_mask, return_classes=False):
        c = self.cond(emb, cls_mask)                                             # (B, K, d)
        m = cls_mask[..., None].float()
        pooled = (c * m).sum(1) / m.sum(1).clamp_min(1)                         # (B, d)
        h = self.pos_proj(self.pos)[None] + pooled[:, None]                      # (B, T, d)
        is_head = (self.slots >= 0)[None, :, None]
        h = h + torch.where(is_head, c[:, self.slots.clamp_min(0)], torch.zeros((), device=h.device))
        h = self.backbone(h)
        h = h + self.xattn(self.xnorm(h), c, c, need_weights=False)[0]
        return (self.proto_out(h), c) if return_classes else self.proto_out(h)

    def context(self, x_t, step, proto, classes):
        """Joint mode: refine prototypes with the current noisy tokens of the whole network.
        x_t (B,T,D), step (B,), proto (B,T,d), classes (B,K,d) -> (B,T,d)."""
        h = self.joint_in(x_t * self.valid) + proto + self.joint_time(sincos(step, self.config["d_model"]))[:, None]
        return proto + self.joint_out(self.joint(h, classes))

    def loss(self, x0, valid, emb, cls_mask, row_valid, tokens_per_step=None, min_snr=0.0, head_weight=1.0):
        """x0 (B,T,D) normalised tokens, valid (T,D), row_valid (B,T) -> denoising MSE.
        min_snr > 0 applies Min-SNR-gamma weighting (Hang et al., 2023), which down-weights the very
        noisy timesteps where the target is unpredictable and speeds up convergence.
        Classifier-row tokens are always included and weighted by head_weight: in deep nets they are a
        tiny fraction of all tokens (10 of 364 in ResNet-20), yet they carry all the class information."""
        joint = self.config.get("joint_layers", 0) > 0
        proto, classes = self.prototypes(emb, cls_mask, return_classes=True)
        if joint:
            B = x0.shape[0]
            step_b = torch.randint(0, len(self.alpha_bar), (B,), device=x0.device)
            ab_b = self.alpha_bar[step_b][:, None, None]
            noise_all = torch.randn_like(x0)
            xt_all = ab_b.sqrt() * x0 + (1 - ab_b).sqrt() * noise_all
            proto = self.context(xt_all, step_b, proto, classes)
        sel = row_valid.nonzero()
        is_head = self.slots[sel[:, 1]] >= 0
        if tokens_per_step and len(sel) > tokens_per_step:
            head, body = sel[is_head], sel[~is_head]
            keep = max(0, tokens_per_step - len(head))
            body = body[torch.randperm(len(body), device=sel.device)[:keep]]
            sel = torch.cat([head, body])
            is_head = self.slots[sel[:, 1]] >= 0
        b, t = sel[:, 0], sel[:, 1]
        x, p, vm = x0[b, t], proto[b, t], valid[t].float()
        if joint:
            step, noise, xt = step_b[b], noise_all[b, t], xt_all[b, t]
            ab = self.alpha_bar[step][:, None]
        else:
            step = torch.randint(0, len(self.alpha_bar), (len(x),), device=x.device)
            ab = self.alpha_bar[step][:, None]
            noise = torch.randn_like(x)
            xt = ab.sqrt() * x + (1 - ab).sqrt() * noise
        pred = self.denoiser(xt, step, p)
        target = x if self.config["pred"] == "x0" else noise
        err = (((pred.float() - target) ** 2) * vm).sum(-1)
        if min_snr > 0:
            snr = (ab / (1 - ab)).squeeze(-1)
            w = snr.clamp(max=min_snr) / min_snr          # x0-prediction form
            if self.config["pred"] != "x0":
                w = w * min_snr / snr.clamp(min=1e-8)     # eps-prediction form: min(snr, g) / snr
            err = err * w
        if head_weight != 1.0:
            hw = torch.where(is_head, torch.full_like(err, head_weight), torch.ones_like(err))
            return (err * hw).sum() / (vm.sum(-1) * hw).sum()
        return err.sum() / vm.sum()

    @torch.no_grad()
    def sample(self, emb, cls_mask, steps=100, chunk=4096, clip="auto", amp=False, body_noise=1.0):
        """DDIM (eta=0). Returns (B, T, D) normalised tokens. `clip` bounds the x0 estimate; "auto" uses
        the largest normalised value seen in the zoo (without it, eps-prediction diverges at t~T)."""
        if clip == "auto":
            clip = self.config.get("clip")
        if amp and emb.is_cuda:  # fp16 inference: tensor cores on T4-class GPUs
            with torch.autocast("cuda", dtype=torch.float16):
                return self.sample(emb, cls_mask, steps, chunk, clip, amp=False, body_noise=body_noise).float()
        joint = self.config.get("joint_layers", 0) > 0
        proto, classes = self.prototypes(emb, cls_mask, return_classes=True)
        B, T, d = proto.shape
        p = proto.reshape(B * T, d)
        x = torch.randn(B * T, self.config["token_size"], device=p.device)
        if body_noise != 1.0:  # body_noise=0 starts the body at the mean: its conditional-mean estimate
            x[(self.slots < 0).repeat(B)] *= body_noise
        ts = torch.linspace(len(self.alpha_bar) - 1, 0, steps, device=p.device).round().long()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=p.device)
            if joint:  # the whole network's current noisy state refines every token's condition
                p = self.context(x.view(B, T, -1), t.expand(B), proto, classes).reshape(B * T, d)
            out = torch.cat([self.denoiser(x[j:j + chunk], t.expand(len(x[j:j + chunk])), p[j:j + chunk]).float()
                             for j in range(0, len(x), chunk)])
            if self.config.get("pred", "eps") == "x0":
                x0 = out
            else:
                x0 = (x - (1 - ab).sqrt() * out) / ab.sqrt()
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.view(B, T, -1)
