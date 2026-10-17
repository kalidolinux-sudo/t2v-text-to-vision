"""Text-conditioned recurrent parameter diffusion: Tina's conditioning on RPG's scalable generator.

  class names --CLIP--> class tokens --set encoder--> condition
  token positions (layer, chunk) + condition + per-row class embedding
      --LSTM (RPG's most accurate recurrent model)--> one prototype per parameter token
  every token is then denoised independently by a small MLP conditioned on its prototype, predicting the
  clean weights (x0, Tina's signal prediction), so memory grows with the number of tokens processed per
  step, not with the model size.

The joint transformer (joint_layers > 0) lets the noisy tokens of one network attend to each other at every
denoising step. joint_arch="tina" follows Tina (Li et al., 2025): one transformer, self-attention only, over
the sequence [timestep token, class tokens, parameter tokens]. joint_arch="decoder" is T2V-1K's decoder
(time added to the input, cross-attention to the classes).
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
    """Residual LSTM over the token sequence (RPG's recurrent model)."""

    def __init__(self, d, layers):
        super().__init__()
        self.net = nn.LSTM(d, d, layers, batch_first=True)

    def forward(self, h):
        return h + self.net(h)[0]


class MLPDenoiser(nn.Module):
    """Residual MLP over one parameter token with FiLM conditioning on (prototype + timestep)."""

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
    def __init__(self, layer_ids, intra_ids, slots, token_size, clip_dim=512, d_model=512, backbone_layers=2,
                 cond_layers=2, heads=8, diffusion_steps=1000, mlp_hidden=512, mlp_depth=3, clip=None,
                 joint_layers=0, valid=None, joint_arch="decoder", **legacy):
        super().__init__()
        # keys of older configs: only the variants that were kept are accepted
        for key, kept in (("backbone", "lstm"), ("denoiser", "mlp"), ("pred", "x0")):
            if legacy.pop(key, kept) != kept:
                raise ValueError(f"{key} must be {kept!r} (other variants were removed)")
        legacy.pop("channels", None)  # conv denoiser widths
        if legacy:
            raise TypeError(f"unexpected arguments: {sorted(legacy)}")
        self.config = dict(token_size=token_size, clip_dim=clip_dim, d_model=d_model,
                           backbone_layers=backbone_layers, cond_layers=cond_layers, heads=heads,
                           diffusion_steps=diffusion_steps, mlp_hidden=mlp_hidden, mlp_depth=mlp_depth, clip=clip,
                           joint_layers=joint_layers, joint_arch=joint_arch)
        self.register_buffer("slots", slots.clone(), persistent=False)
        half = d_model // 2
        self.register_buffer("pos", torch.cat([sincos(layer_ids, half), sincos(intra_ids, half)], -1), persistent=False)
        self.register_buffer("alpha_bar", cosine_alpha_bar(diffusion_steps), persistent=False)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.cond = CondEncoder(clip_dim, d_model, cond_layers, heads)
        self.backbone = Backbone(d_model, backbone_layers)
        self.xattn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.xnorm = nn.LayerNorm(d_model)
        self.proto_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(),
                                       nn.Linear(d_model, d_model))
        self.denoiser = MLPDenoiser(token_size, d_model, mlp_hidden, mlp_depth)
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
            if joint_arch == "tina":
                layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0, activation="gelu",
                                                   batch_first=True, norm_first=True)
                self.joint = nn.TransformerEncoder(layer, joint_layers, norm=nn.LayerNorm(d_model),
                                                   enable_nested_tensor=False)
            elif joint_arch == "decoder":
                layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, dropout=0.0, batch_first=True,
                                                   norm_first=True)
                self.joint = nn.TransformerDecoder(layer, joint_layers)  # self-attention over tokens + cross-attention to classes
            else:
                raise ValueError(joint_arch)
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
        t_emb = self.joint_time(sincos(step, self.config["d_model"]))
        if self.config.get("joint_arch", "decoder") == "decoder":
            h = self.joint_in(x_t * self.valid) + proto + t_emb[:, None]
            return proto + self.joint_out(self.joint(h, classes))
        K = classes.shape[1]
        z = torch.cat([t_emb[:, None], classes, self.joint_in(x_t * self.valid) + proto], 1)
        return proto + self.joint_out(self.joint(z)[:, 1 + K:])

    def loss(self, x0, valid, emb, cls_mask, row_valid, tokens_per_step=None, head_weight=1.0):
        """x0 (B,T,D) normalised tokens, valid (T,D), row_valid (B,T) -> MSE of the clean-weight prediction.
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
            step, xt = step_b[b], xt_all[b, t]
            ab = self.alpha_bar[step][:, None]
        else:
            step = torch.randint(0, len(self.alpha_bar), (len(x),), device=x.device)
            ab = self.alpha_bar[step][:, None]
            noise = torch.randn_like(x)
            xt = ab.sqrt() * x + (1 - ab).sqrt() * noise
        err = (((self.denoiser(xt, step, p).float() - x) ** 2) * vm).sum(-1)
        if head_weight != 1.0:
            hw = torch.where(is_head, torch.full_like(err, head_weight), torch.ones_like(err))
            return (err * hw).sum() / (vm.sum(-1) * hw).sum()
        return err.sum() / vm.sum()

    @torch.no_grad()
    def sample(self, emb, cls_mask, steps=100, chunk=4096, clip="auto", amp=False, body_noise=1.0):
        """DDIM (eta=0). Returns (B, T, D) normalised tokens. `clip` bounds the x0 estimate; "auto" uses
        the largest normalised value seen in the zoo."""
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
            x0 = torch.cat([self.denoiser(x[j:j + chunk], t.expand(len(x[j:j + chunk])), p[j:j + chunk]).float()
                            for j in range(0, len(x), chunk)])
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.view(B, T, -1)
