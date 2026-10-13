"""Frozen CLIP text encoder: one embedding per class name (Tina's conditioning)."""
import torch
import torch.nn.functional as F

CLIP_ID = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo of a {}."
_clip = {}


def _load(device):
    if device not in _clip:
        from transformers import CLIPModel, CLIPTokenizer
        model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        _clip[device] = (model, CLIPTokenizer.from_pretrained(CLIP_ID))
    return _clip[device]


@torch.no_grad()
def embed_names(names, device="cpu"):
    model, tok = _load(device)
    texts = [TEMPLATE.format(n.replace("_", " ")) for n in names]
    batch = tok(texts, padding=True, return_tensors="pt").to(device)
    out = model.get_text_features(**batch)
    if not torch.is_tensor(out):  # newer transformers return a model output object
        out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
    return F.normalize(out.float(), dim=-1)
