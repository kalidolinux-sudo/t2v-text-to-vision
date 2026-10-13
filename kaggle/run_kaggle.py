"""T2V on a free Kaggle GPU: ResNet-20 (272K params, ~270 RPG tokens) end to end.

Needs the t2v/ package next to this file (or attached as a Kaggle dataset) and internet access
(CIFAR-100 and CLIP come from the Hugging Face Hub). Results land in /kaggle/working.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
ZOO = "/tmp/t2v_zoo"           # large; kept out of the saved output
N_TRAIN, N_EVAL = 1000, 60

# locate the t2v package (dataset input, repo checkout, or alongside this script)
if not os.path.isdir(os.path.join(WORK, "t2v")):
    hits = glob.glob("/kaggle/input/**/zoo.py", recursive=True) + glob.glob("**/t2v/zoo.py", recursive=True)
    src = os.path.dirname(hits[0])
    shutil.copytree(src, os.path.join(WORK, "t2v"), ignore=shutil.ignore_patterns("*.json"))
os.chdir(WORK)


def run(cmd):
    print(f"\n>> {cmd}", flush=True)
    t0 = time.time()
    code = subprocess.run(cmd, shell=True).returncode
    print(f"<< exit {code} after {time.time() - t0:.0f}s", flush=True)
    if code:
        sys.exit(code)


run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip install -q transformers huggingface_hub pyarrow")
run(f"python -m t2v.zoo generic --arch resnet20 --out {ZOO} --epochs 40 --lr 0.1")
shards = " & ".join(f"python -m t2v.zoo pmodels --out {ZOO} --split train --n {N_TRAIN} --epochs 5 --min_k 2 "
                    f"--shard {s} --num_shards 4 --fp16 --save_every 50 > {WORK}/zoo_train_{s}.log 2>&1" for s in range(4))
run(f"({shards} & wait)")
run(f"tail -n 1 {WORK}/zoo_train_*.log")
run(f"python -m t2v.zoo pmodels --out {ZOO} --split eval_seen --n {N_EVAL} --n_train_ref {N_TRAIN} --epochs 5 --min_k 2 --fp16")
run(f"python -m t2v.zoo pmodels --out {ZOO} --split eval_unseen --n {N_EVAL} --epochs 5 --fp16")
run(f"python -m t2v.zoo pmodels --out {ZOO} --split eval_novel --n {N_EVAL} --epochs 5 --fp16")
# RPG-style generator: 1024-wide tokens, LSTM backbone, 1-D conv denoiser, checkpoint every 1000 steps
run(f"python -m t2v.train_gen --zoo {ZOO} --out {WORK}/t2v_resnet20.pt --token_size 1024 --d_model 512 "
    f"--denoiser conv --channels 64,128,256 --bs 32 --tokens_per_step 8192 --steps 10000 --warmup 500 "
    f"--pred x0 --prompt_ensemble --emb_noise 0.02 --ema 0.999 --cosine --log_every 250 --save_every 1000")
run(f"python -m t2v.evaluate --ckpt {WORK}/t2v_resnet20.pt --zoo {ZOO} --max_tasks {N_EVAL} --samples 1 "
    f"--steps 20 --out {WORK}/eval_resnet20.json")
