"""Gradio demo: describe a classifier in words, get a ready-to-use vision model.

    python app.py --ckpt ckpt/generator.pt
"""
import argparse
import os
import tempfile
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image

from t2v.generate import build_model, export, generate_flat, load_generator
from t2v.llm import parse_classes
from t2v.zoo import pick_device


RELEASE_URL = "https://github.com/kalidolinux-sudo/t2v-text-to-vision/releases/download/1.0/t2v_1k.zip"
FALLBACK = "checkpoints/t2v_tinacnn_cifar100.pt"


def download(url, path):
    """Fetch a checkpoint (.pt, or a .zip holding one) to `path`."""
    import requests
    import zipfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    print(f"downloading {url} ...", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if url.endswith(".zip"):
        with zipfile.ZipFile(tmp) as z:
            member = next(m for m in z.namelist() if m.endswith(".pt"))
            with z.open(member) as src, open(path, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        os.remove(tmp)
    else:
        os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("T2V_CKPT", "checkpoints/t2v_1k.pt"))
    p.add_argument("--hf_repo", default=os.environ.get("T2V_HF_REPO"),
                   help="Hugging Face model repo to download --ckpt from when it is not on disk (e.g. user/t2v-1k)")
    p.add_argument("--ckpt_url", default=os.environ.get("T2V_CKPT_URL", RELEASE_URL),
                   help="URL (.pt or .zip) to download --ckpt from when it is not on disk; default: the T2V-1K release")
    p.add_argument("--device", default="auto")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    dev = pick_device(a.device)
    if not os.path.exists(a.ckpt) and a.hf_repo:
        from huggingface_hub import hf_hub_download
        a.ckpt = hf_hub_download(a.hf_repo, os.path.basename(a.ckpt))
    if not os.path.exists(a.ckpt) and a.ckpt_url:
        try:
            download(a.ckpt_url, a.ckpt)
        except Exception as e:  # offline: fall back to the small generator shipped in the repository
            print(f"could not download {a.ckpt_url} ({e})")
    if not os.path.exists(a.ckpt):
        if not os.path.exists(FALLBACK):
            raise SystemExit(f"{a.ckpt} not found: download it from the repository's Releases into checkpoints/")
        print(f"{a.ckpt} not available, using {FALLBACK}")
        a.ckpt = FALLBACK
    gen, tok, ck = load_generator(a.ckpt, dev)
    agent = "AMD TokenFactory" if os.environ.get("TOKENFACTORY_API_KEY") else "offline (no TokenFactory key set: write English class names, e.g. \"cat, dog\")"
    state = {}

    def understand(prompt):
        classes, info = parse_classes(prompt, tok.K)
        return ", ".join(classes), f"parsed by: {info['source']}" + (f" — {info['task']}" if info.get("task") else "")

    def create(class_text, steps):
        classes = [c.strip() for c in class_text.split(",") if c.strip()][:tok.K]
        if len(classes) < 2:
            raise gr.Error("Need at least 2 classes.")
        t0 = time.time()
        info = {}
        flat = generate_flat(gen, tok, classes, steps=int(steps), body_noise=0.0, gate="auto", info=info)[0]
        classes = info.get("names", classes)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        model = build_model(ck["arch"], flat, len(classes))
        out = tempfile.mkdtemp(prefix="t2v_")
        onnx_path = export(model, classes, ck["arch"], out)
        state.update(model=model, classes=classes)
        files = [os.path.join(out, "weights.pt")] + ([onnx_path] if onnx_path else [])
        msg = (f"**{tok.num_params:,} parameters** of a `{ck['arch']}` generated in **{dt:.2f}s** "
               f"on `{dev}` — no training, no images.\n\n| class | status | what the agent used |\n|---|---|---|\n")
        known, analogies, descs = info.get("known", {}), info.get("analogies", {}), info.get("descriptions") or {}
        for c in classes:
            status = "known" if known.get(c, True) else "**new**"
            used = []
            if c in analogies:
                used.append("looks like " + ", ".join(f"{n} ({w:.0%})" for n, w in analogies[c].items()))
            if descs.get(c):
                used.append(f"“{descs[c][0]}”")
            msg += f"| {c} | {status} | {' · '.join(used) or '—'} |\n"
        return msg, files

    def predict(img):
        if "model" not in state:
            raise gr.Error("Generate a model first.")
        prob = state["model"].predict_image(np.asarray(Image.fromarray(img).convert("RGB")))
        return {c: float(p) for c, p in zip(state["classes"], prob)}

    with gr.Blocks(title="T2V · Text to Vision") as demo:
        gr.Markdown("# T2V · Text to Vision\nDescribe a classifier in any language. An agent on AMD TokenFactory "
                    "understands the request, describes every class visually and reasons about visual analogies for "
                    "classes it has never seen; a diffusion model then writes every weight of a vision network "
                    f"directly. No training and no images per model.\n\nAgent: **{agent}** · generator: "
                    f"`{ck['arch']}`, {len(ck['names'])} classes in its vocabulary")
        with gr.Row():
            prompt = gr.Textbox(label="What should the model recognise?",
                                placeholder="أبي نموذج يميز بين القطط والكلاب والخيول / sort fruit: apple, orange, pear")
            parse = gr.Button("1 · Understand")
        with gr.Row():
            class_box = gr.Textbox(label="Classes (editable, comma-separated, max %d)" % tok.K)
            parse_info = gr.Markdown()
        steps = gr.Slider(1, 50, value=5, step=1, label="Diffusion steps")
        make = gr.Button("2 · Generate model", variant="primary")
        result = gr.Markdown()
        files = gr.Files(label="Download")
        with gr.Row():
            img = gr.Image(label="3 · Try it on an image")
            pred = gr.Label(num_top_classes=5)
        parse.click(understand, prompt, [class_box, parse_info])
        make.click(create, [class_box, steps], [result, files])
        img.change(predict, img, pred)
    demo.launch(server_name="0.0.0.0", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
