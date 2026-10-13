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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/generator.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    dev = pick_device(a.device)
    gen, tok, ck = load_generator(a.ckpt, dev)
    state = {}

    def understand(prompt):
        classes, info = parse_classes(prompt, tok.K)
        return ", ".join(classes), f"parsed by: {info['source']}" + (f" — {info['task']}" if info.get("task") else "")

    def create(class_text, steps):
        classes = [c.strip() for c in class_text.split(",") if c.strip()][:tok.K]
        if len(classes) < 2:
            raise gr.Error("Need at least 2 classes.")
        t0 = time.time()
        flat = generate_flat(gen, tok, classes, steps=int(steps))[0]
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        model = build_model(ck["arch"], flat, len(classes))
        out = tempfile.mkdtemp(prefix="t2v_")
        onnx_path = export(model, classes, ck["arch"], out)
        state.update(model=model, classes=classes)
        files = [os.path.join(out, "weights.pt")] + ([onnx_path] if onnx_path else [])
        msg = (f"**{tok.num_params:,} parameters** of a `{ck['arch']}` generated in **{dt:.2f}s** "
               f"on `{dev}` — no training, no images.\n\nClasses: {', '.join(classes)}")
        return msg, files

    def predict(img):
        if "model" not in state:
            raise gr.Error("Generate a model first.")
        x = np.asarray(Image.fromarray(img).convert("RGB").resize((32, 32), Image.BICUBIC), dtype=np.float32) / 255
        with torch.no_grad():
            prob = state["model"](torch.from_numpy(x).permute(2, 0, 1)[None]).softmax(-1)[0]
        return {c: float(p) for c, p in zip(state["classes"], prob)}

    with gr.Blocks(title="T2V · Text to Vision") as demo:
        gr.Markdown("# T2V · Text to Vision\nDescribe a classifier in any language. A diffusion model writes every "
                    "weight of a vision network directly — trained once on AMD Instinct, then no training per model.")
        with gr.Row():
            prompt = gr.Textbox(label="What should the model recognise?",
                                placeholder="أبي نموذج يميز بين القطط والكلاب والخيول / sort fruit: apple, orange, pear")
            parse = gr.Button("1 · Understand")
        with gr.Row():
            class_box = gr.Textbox(label="Classes (editable, comma-separated, max %d)" % tok.K)
            parse_info = gr.Markdown()
        steps = gr.Slider(10, 200, value=100, step=10, label="Diffusion steps")
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
