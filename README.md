# T2V · Text to Vision

**Describe a classifier in words, in any language, and get a working vision model: every weight is
written directly by a diffusion model. The model is never trained, and you never upload an image.**

T2V combines two lines of research on parameter generation:

- **Tina** ([Li et al., 2024](https://arxiv.org/abs/2405.14132)): text-conditioned parameter
  diffusion, a "train once, generate for all" model zoo, and classifier-permutation augmentation.
- **RPG** ([Wang et al., 2025](https://arxiv.org/abs/2501.11587)): layer-wise tokenization, a
  recurrent backbone, and per-token denoising, which scales generation to hundreds of millions of
  parameters.

An LLM served on **AMD TokenFactory** turns a free-form request (Arabic, English, …) into class names.
The generator trains on **AMD Instinct** GPUs with ROCm.

```
"أبي نموذج يميز بين القطط والكلاب والخيول"
        │  LLM on AMD TokenFactory  →  ["cat", "dog", "horse"]
        ▼
  CLIP text encoder (frozen)  →  one token per class
        ▼
  set encoder ── recurrent backbone over parameter-token positions ── cross-attention to classes
        ▼                                   (RPG prototypes, Tina conditioning)
  per-token denoiser, DDIM  →  every weight of the network
        ▼
  weights.pt + model.onnx   (no training, no images)
```

## What is new compared with Tina and RPG

| | Tina | RPG | **T2V** |
|---|---|---|---|
| Condition | class names (CLIP) | one-hot task code / LLM embedding | **free-form request in any language → LLM → CLIP** |
| Classifier rows | permutation augmentation | — | **each row token is tied to its class's text embedding**, so class permutation is just a permutation of tokens |
| Scaling | flattened vector (~5K params) | layer-wise tokens + recurrent model (~200M) | RPG tokenization with Tina conditioning |
| Hardware | 2× RTX 8000 | H100 | **AMD Instinct MI300X** (192 GB) + TokenFactory |

## Results so far

These come from CPU-only runs (4 cores) with a Tina-size CNN of 5,418 parameters, every one of them
generated. The zoo is 1,000 CIFAR-100 tasks with 2 to 10 classes each, built from 80 classes; 20
classes are never seen by the zoo. Each test set has 59 or 60 tasks the zoo never saw.

| Test tasks | **T2V generated** | Oracle (fine-tuned on the task) | Classifier selection | Chance |
|---|---|---|---|---|
| New combinations of seen classes | **72.4%** | 73.8% | 72.9% | 20.5% |
| Half of the classes never seen | **49.9%** | 63.8% | n/a | 10.0% |
| All classes never seen (Tina's setting) | **36.7%** | 64.3% | n/a | 10.0% |

Tina reports 30.9% for 100% unseen classes on CIFAR-100. Generating a model takes **0.09 s** on a
laptop-class CPU.

The bugs found and the ablations along the way are recorded in `runs/` (git-ignored).

| Run | Change | Seen | Half unseen |
|---|---|---|---|
| A | ε-prediction | 51.4% | 32.5% |
| **B** | **x₀-prediction (Tina/G.pt signal prediction)** | **72.6%** | **48.3%** |
| C | B + body weights normalised as a deviation from the zoo's mean model | 68.5% | 43.2% |
| D | B + CLIP prompt ensembling + class-embedding noise (σ=0.02) | 72.6% | 49.5% |
| **F** | **D + EMA (0.999) + cosine LR decay** | **72.4%** | **49.9%** |
| G | F on a 3× larger zoo (3,000 tasks), 10k steps | 72.6% | 48.6% |

On fully unseen classes: B 33.0% → D 35.4% → F 36.7%.

G is no better than F (34.4% on fully unseen classes). More tasks built from the same 80 classes add no
new vocabulary. Generalising to new classes needs **more distinct classes** (for example ImageNet-1k
on MI300X), not more combinations of the old ones. The number of sampling steps barely matters: 20
DDIM steps match 100 (49.8% vs 50.0%) and take **0.06 s per model**.

**End-to-end check.** The request "أبي نموذج يميز بين التفاح والبرتقال والدراجات والقروش" goes through
DeepSeek-V4-Flash on TokenFactory and becomes `apples, oranges, bicycles, sharks`. The generated model
(ONNX, valid) scores **85.0%** on the 400 real CIFAR-100 test images of those classes; chance is 25%.

The best generator (run F) is included as `checkpoints/t2v_tinacnn_cifar100.pt` (30 MB), so
`python app.py` works straight after cloning.

With ε-prediction, the first DDIM step divides by √ᾱ_T ≈ 5·10⁻⁵ and the generated weights blow up
by about 800×. The sampler now clips the x₀ estimate to the zoo's normalised range and uses
x₀-prediction by default.

## Run it

```bash
pip install -r requirements.txt     # on AMD Developer Cloud, ROCm PyTorch is usually preinstalled

# 1. model zoo (Tina recipe); shards run in parallel on one MI300X
python -m t2v.zoo generic --arch resnet20 --out zoo
for s in 0 1 2 3; do python -m t2v.zoo pmodels --out zoo --n 10000 --shard $s --num_shards 4 & done; wait
python -m t2v.zoo pmodels --out zoo --split eval_seen  --n 100
python -m t2v.zoo pmodels --out zoo --split eval_unseen --n 100
python -m t2v.zoo pmodels --out zoo --split eval_novel  --n 100

# 2. generator (RPG-style), bf16 on MI300X
python -m t2v.train_gen --zoo zoo --out ckpt/generator.pt --bf16 --token_size 1024 --steps 50000

# 3. evaluate against Tina's baselines
python -m t2v.evaluate --ckpt ckpt/generator.pt --zoo zoo

# 4. text → model
export TOKENFACTORY_API_KEY=...            # never commit it
export TOKENFACTORY_BASE_URL=https://developer.amd.com.cn/radeon/api/v1
export TOKENFACTORY_MODEL=DeepSeek-V4-Flash
python -m t2v.generate --prompt "أبي نموذج يميز بين القطط والكلاب والخيول"
python app.py                              # Gradio demo
```

`scripts/smoke_test.sh` runs the whole pipeline at toy scale on a CPU in about 2 minutes.

Target architectures: `tina_cnn` (5.4K params), `resnet20` (0.27M), `resnet18` (11M).

## Limitations

- The generator itself has to be trained, once, on a model zoo. The "no training" claim applies to
  each new model, not to the system as a whole.
- Accuracy drops on classes far from the zoo's distribution. Unseen classes work through CLIP's text
  space, but they work less well than seen ones.
- The CPU results above use a 5K-parameter network. Scaling to ResNet-18 (11M parameters) is the
  next step on MI300X.
