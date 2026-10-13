"""Frozen CLIP text encoder: one embedding per class name (Tina's conditioning)."""
import torch
import torch.nn.functional as F

CLIP_ID = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo of a {}."
# CLIP's prompt-ensembling trick: averaging several templates gives steadier class embeddings.
TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a close-up photo of a {}.", "a low resolution photo of a {}.",
             "a bright photo of a {}.", "a cropped photo of a {}.", "a photo of the small {}.", "a photo of the large {}."]
_clip = {}


def _load(device):
    if device not in _clip:
        from transformers import CLIPModel, CLIPTokenizer
        model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        _clip[device] = (model, CLIPTokenizer.from_pretrained(CLIP_ID))
    return _clip[device]


@torch.no_grad()
def embed_names(names, device="cpu", ensemble=False):
    model, tok = _load(device)
    templates = TEMPLATES if ensemble else [TEMPLATE]
    embs = []
    for template in templates:
        texts = [template.format(n.replace("_", " ")) for n in names]
        batch = tok(texts, padding=True, return_tensors="pt").to(device)
        out = model.get_text_features(**batch)
        if not torch.is_tensor(out):  # newer transformers return a model output object
            out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
        embs.append(F.normalize(out.float(), dim=-1))
    return F.normalize(torch.stack(embs).mean(0), dim=-1)
