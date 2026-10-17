#!/usr/bin/env bash
# End-to-end check of the whole pipeline at toy scale (CPU is fine, ~10 min). Numbers are meaningless.
set -euo pipefail
cd "$(dirname "$0")/.."; export DATA=${DATA:-data}
Z=runs/smoke/zoo; C=runs/smoke/gen.pt
python -m t2v.zoo generic --arch tina_cnn --out $Z --epochs 2
python -m t2v.zoo pmodels --out $Z --split train --n 24 --epochs 1 --min_k 6
python -m t2v.zoo pmodels --out $Z --split eval_seen --n 4 --n_train_ref 24 --epochs 1 --min_k 6
python -m t2v.zoo pmodels --out $Z --split eval_unseen --n 4 --epochs 1
python -m t2v.train_gen --zoo $Z --out $C --token_size 512 --d_model 128 --mlp_hidden 256 --steps 60 \
  --bs 8 --tokens_per_step 512 --log_every 20 --save_every 1000
python -m t2v.evaluate --ckpt $C --zoo $Z --max_tasks 2 --samples 2 --steps 10 --out runs/smoke/eval.json
python -m t2v.generate --ckpt $C --prompt "apple, orange, pear, tiger" --steps 10 --out runs/smoke/generated
