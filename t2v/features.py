"""Frozen CLIP ViT-B/32 image features: image prompts (t2v.generate --image_dir) and the distillation targets
of the generic model (t2v.zoo generic --clip_distill)."""
import torch
import torch.nn.functional as F

from .text_embed import _load

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_pixels(x_uint8, size=224):
    """uint8 (N,3,H,W) -> CLIP-normalised float (N,3,224,224)."""
    x = F.interpolate(x_uint8.float() / 255, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)


@torch.no_grad()
def image_features(x_uint8, device="cpu", bs=256):
    """uint8 images (N,3,H,W) -> unit-norm CLIP image embeddings (N,512) on the CPU (fp16 autocast on GPU)."""
    model, _ = _load(device)
    out = []
    for i in range(0, len(x_uint8), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            f = model.get_image_features(pixel_values=clip_pixels(x_uint8[i:i + bs].to(device)))
        if not torch.is_tensor(f):
            f = f.pooler_output if getattr(f, "pooler_output", None) is not None else f[0]
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)
