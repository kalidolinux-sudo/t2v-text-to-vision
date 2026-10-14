# T2V · Text to Vision

**Describe a classifier in words, in any language, and get a working vision model in a fraction of a
second. An agent on AMD TokenFactory understands the request and reasons about what each class looks
like; a diffusion model then writes every weight of the network directly. No training and no images
per model.**

```
"أبي نموذج يميز بين الكوالا والبانجو والنعامة"            (any language)
        │
        ▼  agent on AMD TokenFactory (DeepSeek-V4-Flash)
  1 understand   → ["koala", "banjo", "ostrich"]
  2 describe     → "koala: Gray, fluffy, tree-dwelling marsupial with large nose and round ears.", …
  3 analogise    → koala never seen?  looks like teddy 50%, wombat 30%, wallaby 20%
        │
        ▼  T2V generator (Tina × RPG parameter diffusion, 49M params)
  CLIP text condition → layer-wise parameter tokens → 5 DDIM steps
        │
        ▼
  ResNet-20 with all 274,042 weights generated  →  weights.pt / model.onnx   (0.08 s on a T4)
```

## T2V-1K: results on ImageNet-1k

ImageNet-1k at 32×32. The zoo uses **900 classes**. **100 classes are never seen**: 49 drawn at random,
and 51 forming whole semantic groups (sharks and turtles, musical instruments, …) so that no close
relative is left among the seen classes. Each row is 100 test tasks of 10 classes that the zoo never saw.
The oracle is a ResNet-20 actually fine-tuned on the images of each task.

| Test tasks (100 each, 10 classes) | **T2V-1K, generated from text** | Oracle (fine-tuned on images) | Share of oracle | Chance |
|---|---|---|---|---|
| Seen classes, new combinations | **70.0%** | 72.4% | 97% | 10% |
| Half the classes never seen | **64.1%** | 74.9% | 86% | 10% |
| All classes never seen | **63.6%** | 74.1% | 86% | 10% |
| All never seen, whole groups held out | **47.5%** | 64.8% | 73% | 10% |

The run's console log and zoo statistics are in [`results/t2v_1k/`](results/t2v_1k/).

One full ResNet-20 takes **0.079 s** to generate (T4, 5 DDIM steps, fp16). Picking rows from the generic
model's head (Tina's "selection" baseline) scores 69.3% on seen classes and cannot handle new classes at
all.

### What the TokenFactory agent adds

| Prior for classifier rows of new classes | Half new | All new | All new, hard |
|---|---|---|---|
| Ridge map from class text only | 64.0% | 61.2% | **48.3%** |
| + CLIP nearest neighbours (no LLM) | 64.2% | 62.9% | 48.1% |
| **+ TokenFactory visual analogies** | 64.1% | **63.6%** | 47.5% |
| No known-class gate | 64.3% | 60.2% | 44.1% |

- The analogies help most when visually close relatives exist among the seen classes (+2.4 points over
  ridge on random new classes). When a whole group is held out, there is little to draw an analogy
  from, and they do not help. `notebooks/t2v_colab_paired_stats.py` gives paired bootstrap intervals for
  these differences.
- The blend weight between the ridge prior and the analogies is chosen by cross-validation on 100 seen
  classes treated as new, never on test tasks.
- The visual descriptions (5 per class for all 1,000 classes) are written by the same agent. On the
  CIFAR-100 ResNet-20 they added 3 to 6 points (development log below).

### What the diagnostics say (`python -m t2v.diagnose`, 50 tasks per split)

- **The model follows the text.** Generated with the wrong class names, accuracy falls to chance
  (9.3%). Reversing the class order gives the same accuracy (69.1% vs 69.3%).
- **The classifier rows are the bottleneck, not the body.** With fine-tuned rows on the generated body,
  new-class accuracy goes from 59.1% to 71.6%. A fine-tuned body under the generated rows adds only 0.5
  to 2 points. The generated body is as close to the right one as the zoo's mean model is.
- **The zoo does carry task-specific body information.** With tail fine-tuning and two-seed model soups,
  task-to-task variation is 27× the seed noise (`zoo analyze`). Most of it is in BatchNorm statistics of
  the last two blocks, and it is worth little accuracy.
- **The known-class gate matters.** It keeps the generated row correction only for classes the
  generator was trained on, and it adds 4.2 points on the hard split. Its threshold (0.9947) is set
  automatically just above the closest pair of different training classes.

### Where AMD comes in

- **AMD TokenFactory** is the agent's reasoning core, in the product and at build time. It parses each
  request, writes visual descriptions of new classes on the fly, and picks visual analogies for names
  it has not seen before. It also wrote the descriptions and analogies shipped in `data/`.
- **Training** of the zoo and the generator ran once, on a Google Colab T4, in 72 minutes
  (`notebooks/t2v_colab_final_imagenet.py`). The code also runs on AMD GPUs with ROCm PyTorch
  (`--bf16` for MI300X, and a ROCm build of the Docker image).

## Quick start

The ImageNet-1k generator `t2v_1k.pt` (198 MB) is attached to the
[**T2V-1K release**](https://github.com/kalidolinux-sudo/t2v-text-to-vision/releases/tag/1.0) as
`t2v_1k.zip`. The app downloads and unpacks it into `checkpoints/` on first start. The small CIFAR-100
generator (`checkpoints/t2v_tinacnn_cifar100.pt`, 30 MB) ships with the code and is used when offline.

```bash
# Docker (CPU); the TokenFactory key is passed at run time and never stored in the image
docker build -t t2v .
docker run -p 7860:7860 -e TOKENFACTORY_API_KEY=... t2v       # fetches t2v_1k.pt from the release
docker run -p 7860:7860 -e T2V_CKPT=checkpoints/t2v_tinacnn_cifar100.pt t2v   # small CIFAR-100 model
# AMD GPU: docker build -t t2v --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4 .
```

Without Docker: `pip install -r requirements.txt && python app.py`. Without a TokenFactory key the app
still works: requests are split on commas and the stored descriptions and analogies are used.

**Reproduce T2V-1K:** paste `notebooks/t2v_colab_final_imagenet.py` into one Google Colab cell (T4 GPU)
and run it. It needs no API key, because the TokenFactory outputs are bundled. It takes about 72
minutes and saves everything to Google Drive, resuming after a disconnect.

## Repository layout

| Path | Contents |
|---|---|
| `t2v/` | the package: data and zoo (`data`, `zoo`), tokenizer and generator, training, generation, evaluation, diagnostics, and the TokenFactory agent (`llm`, `describe`, `neighbors`) |
| `app.py`, `Dockerfile` | Gradio demo, CPU or ROCm container |
| `data/` | ImageNet-1k class names and split, and the TokenFactory outputs (descriptions of 1,000 classes, visual analogies) |
| `notebooks/` | one-cell Colab runs; [`notebooks/README.md`](notebooks/README.md) lists them |
| `results/` | logs and diagnostics of the T2V-1K run |
| `checkpoints/` | the CIFAR-100 generator; the ImageNet-1k one is fetched from the release |

## Development log (CIFAR-100)

The method was developed on CIFAR-100 before scaling to ImageNet-1k. T2V combines two lines of
research on parameter generation:

- **Tina** ([Li et al., 2024](https://arxiv.org/abs/2405.14132)): text-conditioned parameter
  diffusion, a "train once, generate for all" model zoo, and classifier-permutation augmentation.
- **RPG** ([Wang et al., 2025](https://arxiv.org/abs/2501.11587)): layer-wise tokenization, a
  recurrent backbone, and per-token denoising, which scales generation to hundreds of millions of
  parameters.

| | Tina | RPG | **T2V** |
|---|---|---|---|
| Condition | class names (CLIP) | one-hot task code / LLM embedding | **free-form request in any language → LLM agent → CLIP + visual descriptions** |
| Classifier rows | permutation augmentation | — | **each row token is tied to its class's text embedding**; rows = semantic prior + generated correction |
| New classes | CLIP text only | — | **ridge prior blended with LLM visual analogies**, known-class gate |
| Scaling | flattened vector (~5K params) | layer-wise tokens + recurrent model (~200M) | RPG tokenization with Tina conditioning and a joint transformer |

### 1. Tina-size CNN on CIFAR-100 (CPU only)

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
new vocabulary. Generalising to new classes needs **more distinct classes** (ImageNet-1k, see T2V-1K above),
not more combinations of the old ones. The number of sampling steps barely matters: 20
DDIM steps match 100 (49.8% vs 50.0%) and take **0.06 s per model**.

**End-to-end check.** The request "أبي نموذج يميز بين التفاح والبرتقال والدراجات والقروش" goes through
DeepSeek-V4-Flash on TokenFactory and becomes `apples, oranges, bicycles, sharks`. The generated model
(ONNX, valid) scores **85.0%** on the 400 real CIFAR-100 test images of those classes; chance is 25%.

The best generator (run F) is included as `checkpoints/t2v_tinacnn_cifar100.pt` (30 MB), so
`python app.py` works straight after cloning.

With ε-prediction, the first DDIM step divides by √ᾱ_T ≈ 5·10⁻⁵ and the generated weights blow up
by about 800×. The sampler now clips the x₀ estimate to the zoo's normalised range and uses
x₀-prediction by default.

### 2. ResNet-20 on CIFAR-100 (Colab T4)

This run used Google Colab with a T4 GPU. Every one of the 274,042 parameters is generated, about 50×
the CNN above. The zoo has 300 tasks, and the generator trained for 3,000 steps in 7 minutes
(`notebooks/cifar100_dev/t2v_colab_fast.py`, then `notebooks/cifar100_dev/t2v_colab_retrain_v4.py`).

| Test tasks (30 each) | **T2V generated** | Oracle (fine-tuned) | Classifier selection | Chance |
|---|---|---|---|---|
| Seen classes | **73.9%** | 89.3% | 85.9% | 21.5% |
| Half unseen | **49.4%** | 79.8% | n/a | 10.0% |
| All unseen | **33.0%** | 78.3% | n/a | 10.0% |

Generating one full ResNet-20 takes 0.12 s. Tina generates only the ~640-parameter classifier of a
ResNet-20. T2V generates the whole network.

Three things it took to get there, all now options in `t2v.train_gen`:

- **`--norm element`**: body weights are generated as a deviation from the zoo's mean model. With
  per-tensor normalisation the loss plateaued at 0.44. A noise test explains why: ResNet-20 collapses
  to chance once weight errors pass about 2× the zoo's per-weight spread, which per-tensor units
  easily exceed.
- **`--head_weight 10`**: classifier rows are 10 of 364 tokens. With uniform token sampling the body
  came out right (with the true head it matched the oracle), but the generated class rows were nearly
  identical and accuracy sat at chance. Always including the rows and up-weighting them fixed it:
  22% → 73.9%.
- **`--fp16` + MLP denoiser**: 7.6 steps/s on a T4, against 1 step/s with the conv denoiser in fp32.

#### T2V-Hybrid: more of Tina and RPG together

The same ResNet-20 zoo and settings were used for every run below (30 test tasks per row).

| Run | Seen | Half unseen | All unseen | s/model |
|---|---|---|---|---|
| v4: RPG generator, element norm, head weighting | 73.9% | 49.4% | 33.0% | 0.12 |
| v4b: + EMA warm-up fix | 80.3% | 44.5% | 34.0% | 0.12 |
| **v6b: + Tina/G.pt joint transformer + LLM visual descriptions** | **81.2%** | **51.5%** | **41.5%** | 0.23 |
| Oracle (fine-tuned on each task) | 89.3% | 79.8% | 78.3% | n/a |

- **Joint transformer (`--joint_layers 4`)**: this follows Tina and G.pt. At every denoising step, all
  noisy parameter tokens of the network attend to each other and to the class tokens, so the head is
  denoised in step with the body. The output refines RPG's prototypes, and RPG's per-token denoiser
  still decodes each token.
- **Visual descriptions (`--descriptions`, `t2v.describe`)**: this follows CuPL. DeepSeek on AMD
  TokenFactory writes five visual descriptions per class, and the CLIP condition averages them with the
  prompt templates. At generation time the request parser writes task-aware descriptions, for example
  "ripe banana: yellow peel with brown spots" against "unripe banana: green peel". Classes the
  generator never saw are described on the fly.
- **EMA warm-up**: a fixed 0.999 decay kept 37% of the random initial weights after 1,000 steps and 5%
  after 3,000.

Two runs were not separated: the joint transformer and the descriptions were added together. On the
small CNN, descriptions alone gave +0.9 points on fully unseen classes (34.8% → 35.7%, 60 tasks), which
is within noise.

#### Component analysis: fewer sampling steps are better *and* faster

Profiling (`python -m t2v.profile`, ResNet-20 layout) shows that almost all generation time is spent in
the per-step modules. The joint transformer has 37% of the parameters and takes 52% of the time, and
the per-token denoiser has 37% and takes 45%. So the step count is the main speed lever. Sweeping it
on the trained v6b generator (30 tasks per test, T4) gave:

| v6b sampling | Seen | Half unseen | All unseen | s/model |
|---|---|---|---|---|
| 20 steps fp32 | 79.9% | 52.6% | 40.3% | 0.218 |
| 10 steps fp32 | 81.5% | 55.3% | 42.1% | 0.112 |
| **5 steps fp16** | **88.4%** | **60.9%** | **46.7%** | **0.057** |
| Oracle (fine-tuned) | 89.3% | 79.8% | 78.3% | n/a |

At 5 steps the generated ResNet-20 matches a fine-tuned one on seen classes and beats classifier
selection (85.9%), at a quarter of the cost. A likely reason is that the early x0 estimates sit close
to the conditional mean of the weights, and later steps add sample-specific detail that classification
does not need. The default is now 5 steps.

Ablations on the same zoo, at 20 steps:

| Run | Seen | Half unseen | All unseen |
|---|---|---|---|
| v4b | 80.3% | 44.5% | 34.0% |
| + descriptions only | 83.1% | 50.2% | 37.3% |
| + descriptions + joint transformer (v6b) | 79.9% | 52.6% | 40.3% |
| + descriptions + lite joint (2 layers, 1-layer class encoder) | 80.6% | 42.3% | 38.6% |

The descriptions help on every test. The joint transformer helps on unseen classes, and the lite
variant is not worth it.

#### Diagnostics-driven fixes (100 test tasks per row, about ±3 points)

`python -m t2v.diagnose` runs eight inference-only probes. On ResNet-20 they showed three things:
- **Body:** the generated body was noisier than the zoo's mean model (the zoo's bodies differ mainly
  by fine-tuning noise).
- **Classifier rows for unseen classes:** they snap to rows of known classes. For 99.9% of them the
  nearest training row is closer than the correct row.
- **Ridge baseline:** a plain ridge map from class text to rows, with lambda chosen by cross-validation
  over seen classes, beats the generator on unseen classes.

| ResNet-20, 5 steps fp16 | Seen | All unseen |
|---|---|---|
| v6b | 84.0% | 45.9% |
| v6b, noise-free body (`--body_noise 0`) | 86.3% | 46.5% |
| v7: semantic head prior + generated correction (`--semantic_prior`) | 86.2% | 53.5% |
| **v7 + known-class gate (`--gate 0.98`)** | **86.3%** | **58.1%** |
| Ridge prior alone | 85.1% | 57.7% |
| Oracle (fine-tuned) | 87.2% | 78.4% |

With the gate, the generated correction is kept only for classes the generator was trained on (CLIP
similarity ≥ 0.98; different CIFAR classes never exceed 0.954). New classes get the semantic prior,
which generalises, and known classes get the prior plus the generated correction.


## Command-line pipeline

```bash
pip install -r requirements.txt     # on AMD Developer Cloud, ROCm PyTorch is usually preinstalled

# 1. generic model + model zoo (Tina recipe); shards run in parallel on one GPU
python -m t2v.zoo generic --dataset imagenet32 --split_file data/imagenet_split.json --arch resnet20 --out zoo --amp
for s in 0 1 2 3; do python -m t2v.zoo pmodels --out zoo --n 1000 --min_k 2 --finetune tail:2 --soup 2 --amp \
    --shard $s --num_shards 4 & done; wait
for s in eval_seen eval_unseen eval_novel eval_novel_hard; do python -m t2v.zoo pmodels --out zoo --split $s \
    --n 100 --finetune tail:2 --soup 2 --amp; done

# 2. TokenFactory outputs (already in data/ for ImageNet-1k)
export TOKENFACTORY_API_KEY=...            # never commit it
export TOKENFACTORY_BASE_URL=https://developer.amd.com.cn/radeon/api/v1
export TOKENFACTORY_MODEL=DeepSeek-V4-Flash
python -m t2v.describe --names_from zoo/meta.json --out data/descriptions.json
python -m t2v.neighbors --zoo zoo --descriptions data/descriptions.json --out data/neighbors.json

# 3. generator (add --bf16 on MI300X, --fp16 on T4-class GPUs)
python -m t2v.train_gen --zoo zoo --out ckpt/generator.pt --denoiser mlp --mlp_hidden 1024 --joint_layers 4 \
    --norm element --head_weight 10 --semantic_prior --descriptions data/descriptions.json \
    --neighbors data/neighbors.json --prompt_ensemble --emb_noise 0.02 --ema 0.999 --cosine --steps 6000

# 4. evaluate and diagnose
python -m t2v.evaluate --ckpt ckpt/generator.pt --zoo zoo --steps 5 --body_noise 0 --gate auto
python -m t2v.diagnose --ckpt ckpt/generator.pt --zoo zoo --body_noise 0

# 5. text → model
python -m t2v.generate --ckpt ckpt/generator.pt --prompt "أبي نموذج يميز بين القطط والكلاب والخيول"
```

`scripts/smoke_test.sh` runs the whole pipeline at toy scale on a CPU in about 2 minutes. Target
architectures: `tina_cnn` (5.4K params), `resnet20` (0.27M), `resnet18` (11M).

## Limitations

- The generator itself is trained once, on a model zoo. "No training" applies to each new model, not
  to the system as a whole.
- Images are 32×32, so photos are downscaled heavily, and fine-grained classes (dog breeds, for
  example) are hard.
- New classes far from everything seen (whole held-out groups) reach 73% of the fine-tuned oracle, against
  86% for random new classes.
- The generated network body stays close to the zoo's mean model. The task-specific part of a model is
  carried by its classifier rows.

## License

MIT, see `LICENSE`.
