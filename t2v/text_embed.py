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
def _encode(texts, device):
    model, tok = _load(device)
    batch = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    out = model.get_text_features(**batch)
    if not torch.is_tensor(out):  # newer transformers return a model output object
        out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
    return F.normalize(out.float(), dim=-1)


@torch.no_grad()
def embed_names(names, device="cpu", ensemble=False, descriptions=None):
    """One embedding per class: the average over prompt templates and, when `descriptions` has entries
    for a class, over its LLM-written visual descriptions too (CuPL)."""
    templates = TEMPLATES if ensemble else [TEMPLATE]
    embs = []
    for n in names:
        name = n.replace("_", " ")
        texts = [t.format(name) for t in templates]
        texts += [f"{name}: {d}" for d in (descriptions or {}).get(n) or (descriptions or {}).get(name) or []]
        embs.append(F.normalize(_encode(texts, device).mean(0), dim=-1))
    return torch.stack(embs)
