"""Hugging Face Space entry point: fetch the T2V code from GitHub and start its Gradio app.

Settings -> Secrets: TOKENFACTORY_API_KEY (the AMD TokenFactory key). The trained ImageNet-1k generator
is downloaded from the GitHub release on first start.
"""
import os
import runpy
import subprocess
import sys

REPO = "https://github.com/kalidolinux-sudo/t2v-text-to-vision"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2v-src")

os.environ.setdefault("TOKENFACTORY_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
os.environ.setdefault("TOKENFACTORY_MODEL", "DeepSeek-V4-Flash")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

if not os.path.isdir(os.path.join(SRC, "t2v")):
    subprocess.run(["git", "clone", "--depth", "1", REPO, SRC], check=True)
os.chdir(SRC)
sys.path.insert(0, SRC)
sys.argv = ["app.py", "--port", "7860"]
runpy.run_path("app.py", run_name="__main__")
