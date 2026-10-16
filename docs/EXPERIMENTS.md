# T2V experiment log

Every run that shaped T2V, in order: what was tried, what it gave, and what was learned, including the
bugs and the ideas that did not work. All accuracies are mean test accuracy of generated models on tasks
the zoo never saw. With 30 test tasks per row a difference of about ±5 points can be noise; with 100
tasks, about ±3.

Compute: a 4-core CPU for stage 1, a Google Colab T4 for stages 2 and 3. Language model: DeepSeek-V4-Flash
on AMD TokenFactory.

---

## Stage 1 — Tina-size CNN on CIFAR-100, CPU only

Setup: a 5,418-parameter CNN (`tina_cnn`), every weight generated. Zoo of 1,000 tasks with 2 to 10
classes, built from 80 classes; 20 classes are never seen. 59 or 60 test tasks per split.

| Run | Change | Seen | Half unseen | All unseen |
|---|---|---|---|---|
| A | ε-prediction (DDPM default) | 51.4% | 32.5% | — |
| B | **x₀-prediction** (Tina/G.pt signal prediction) | 72.6% | 48.3% | 33.0% |
| C | B + body as deviation from the zoo's mean model | 68.5% | 43.2% | — |
| D | B + CLIP prompt ensembling + class-embedding noise σ=0.02 | 72.6% | 49.5% | 35.4% |
| **F** | **D + EMA 0.999 + cosine LR** | **72.4%** | **49.9%** | **36.7%** |
| G | F on a 3× larger zoo (3,000 tasks), 10k steps | 72.6% | 48.6% | 34.4% |
| — | Oracle (fine-tuned on each task) | 73.8% | 63.8% | 64.3% |

- Tina reports 30.9% for fully unseen CIFAR-100 classes; F reaches 36.7%.
- **Bug: ε-prediction diverged.** At t≈T the first DDIM step divides by √ᾱ_T ≈ 5·10⁻⁵, and the weights
  blew up by about 800×. Fix: clip the x₀ estimate to the zoo's range and predict x₀ directly (A → B).
- **Negative result: more tasks from the same classes do not help (G).** The zoo needs more distinct
  classes, not more combinations of the old ones. This led to ImageNet-1k (stage 3).
- 20 DDIM steps match 100 (49.8% vs 50.0% half unseen), at 0.06 s per model.
- Visual descriptions alone gave +0.9 points on fully unseen classes (34.8% → 35.7%), within noise at
  this size.
- **End-to-end check.** The Arabic request "أبي نموذج يميز بين التفاح والبرتقال والدراجات والقروش" went
  through TokenFactory and became `apples, oranges, bicycles, sharks`. The generated model scored 85.0% on
  the 400 real CIFAR-100 test images of those classes (chance 25%).
- Engineering: the CIFAR-100 download from the original host hung, so data comes from the Hugging Face
  parquet mirror (whose metadata spells one class "cra"; fixed to "crab"). The 1-D conv denoiser was too
  slow on CPU, so an MLP per-token denoiser was added.

## Stage 2 — ResNet-20 on CIFAR-100, Colab T4

Setup: ResNet-20 with all 274,042 parameters generated (about 50× stage 1), a 300-task zoo, 80 seen
classes. Oracle: 89.3% / 79.8% / 78.3% (seen / half unseen / all unseen); classifier selection 85.9%.

### Getting ResNet-20 to work at all (v4)

| Problem found | Fix | Effect |
|---|---|---|
| Generated ResNet-20s were at chance; loss stuck at 0.44 with per-tensor normalisation. A noise test showed ResNet-20 collapses once weight errors pass ~2× the zoo's per-weight spread. | `--norm element`: generate the body as a deviation from the zoo's mean model | body correct |
| The 10 classifier-row tokens are 10 of 364; with uniform token sampling the generated rows were near-identical | `--head_weight 10`, head tokens always included in each step | 22% → 73.9% seen |
| 1 step/s with the conv denoiser in fp32 | MLP denoiser + `--fp16` | 7.6 steps/s |

v4: 73.9% / 49.4% / 33.0% (30 tasks), 0.12 s per model.

### Generator versions (30 test tasks each)

| Run | Change | Seen | Half unseen | All unseen | s/model |
|---|---|---|---|---|---|
| v4 | element norm + head weighting | 73.9% | 49.4% | 33.0% | 0.12 |
| v4b | + EMA warm-up fix | 80.3% | 44.5% | 34.0% | 0.12 |
| v6b | + Tina/G.pt joint transformer (4 layers) + LLM visual descriptions | 81.2% | 51.5% | 41.5% | 0.23 |

- **Bug: EMA kept the random initialisation.** A fixed 0.999 decay still holds 37% of the initial weights
  after 1,000 steps and 5% after 3,000. Fix: warm the decay up as (1+step)/(10+step).
- Min-SNR loss weighting was tested as a suspect for the v4b drop on half-unseen tasks; it was not the
  cause.

Ablation at 20 sampling steps:

| Run | Seen | Half unseen | All unseen |
|---|---|---|---|
| v4b | 80.3% | 44.5% | 34.0% |
| + descriptions only | 83.1% | 50.2% | 37.3% |
| + descriptions + joint transformer (v6b) | 79.9% | 52.6% | 40.3% |
| + descriptions + lite joint (2 layers, 1-layer class encoder) | 80.6% | 42.3% | 38.6% |

Descriptions help on every split (+2.8 to +5.7 points, 30 tasks per split, so each difference is near the noise level on its own). The joint transformer helps on unseen classes. The
lite variant is not worth it.

### Component analysis: fewer sampling steps are better and faster

Profiling (`python -m t2v.profile`): the joint transformer holds 37% of the parameters and takes 52% of
generation time, the per-token denoiser 37% and 45%. So the step count is the speed lever.

| v6b sampling | Seen | Half unseen | All unseen | s/model |
|---|---|---|---|---|
| 20 steps fp32 | 79.9% | 52.6% | 40.3% | 0.218 |
| 10 steps fp32 | 81.5% | 55.3% | 42.1% | 0.112 |
| **5 steps fp16** | **88.4%** | **60.9%** | **46.7%** | **0.057** |

At 5 steps the generated ResNet-20 matches the fine-tuned oracle on seen classes and beats classifier
selection. The early x₀ estimates sit near the conditional mean of the weights; later steps add
sample-specific detail that classification does not need. 5 steps became the default.

### Diagnostics-driven fixes (100 test tasks per row)

`python -m t2v.diagnose` runs eight inference-only probes. On the CIFAR ResNet-20 they showed:

- **Body:** the generated body is essentially the zoo's mean model. In a 2-epoch zoo the bodies differ
  mainly by fine-tuning noise, so there is no task signal for the generator to learn.
- **Rows for unseen classes snap to known classes:** for 99.9% of them the nearest training row is closer
  than the correct row.
- **The head is the bottleneck:** fine-tuned rows on the generated body reach 71.0% on unseen classes.
- **A ridge map from class text to rows beats the generator on unseen classes.**

| ResNet-20, 5 steps fp16 | Seen | All unseen |
|---|---|---|
| v6b | 84.0% | 45.9% |
| v6b, noise-free body (`--body_noise 0`) | 86.3% | 46.5% |
| v7: semantic head prior + generated correction (`--semantic_prior`) | 86.2% | 53.5% |
| **v7 + known-class gate (0.98)** | **86.3%** | **58.1%** |
| Ridge prior alone | 85.1% | 57.7% |
| Oracle | 87.2% | 78.4% |

- **Gate threshold:** 0.95 was too low, because two different CIFAR classes reach 0.954 CLIP similarity;
  0.98 keeps every class separate.
- **Corrected mistakes in the analysis itself:** the ridge λ was first chosen on test tasks; it is now
  chosen by 5-fold cross-validation over seen classes. A first memorisation metric was dominated by body
  noise, which made it look as if the generator copied training models; the metric now works on
  classifier rows only, and that claim was withdrawn.
- Code bugs fixed along the way: an aliasing error when adding prior rows in place (`add_head_rows`), and
  tokenizer statistics left on the CPU when a checkpoint was loaded on the GPU.

## Stage 3 — T2V-1K on ImageNet-1k (the final model)

Setup: ImageNet-1k at 32×32 (Hugging Face `benjamin-paine/imagenet-1k-32x32`, 260 training images per
class, the labelled validation split as test). **900 seen classes; 100 never seen**: 49 random and 51
forming whole CLIP k-means clusters (sharks and turtles, musical instruments, cliff dwelling), so no close
relative stays among the seen classes. One Colab cell, `notebooks/t2v_colab_final_imagenet.py`, 72
minutes on a T4. Console log: `results/t2v_1k/colab_run_log.txt`.

What changed from stage 2, and why:

| Change | Reason |
|---|---|
| 1,000 classes instead of 100 | stage 1 showed that new classes need more distinct classes |
| Tail fine-tuning (`--finetune tail:2`) | the CIFAR body carried no task signal; adapting only the last two blocks concentrates task differences where they exist |
| Model soups (`--soup 2`) | averaging two seeds halves the seed noise in each zoo model |
| `zoo analyze` | measures task signal against seed noise directly |
| TokenFactory role 3: visual analogies | rows of new classes snapped to known classes anyway; an LLM picks which known classes they should resemble |
| Gate set automatically | ImageNet has near-synonyms (closest pair of training classes 0.9897), so 0.98 would call new classes known |
| Mixed precision in zoo training | speed |

### Zoo

| Step | Result | Time |
|---|---|---|
| Generic ResNet-20, 900 classes, 15 epochs | 17.3% top-1 over 900 classes | 7 min |
| 1,000 train tasks (2–10 classes, 4 epochs, tail:2, soup 2) | 80.7% mean accuracy | 12 min |
| 400 test tasks, same recipe (oracle) | seen 72.4%, half new 74.9%, all new 74.1%, hard 64.8% | 10 min |
| Signal-to-noise of the body | **26.8** (task variation 27× seed noise; mostly BatchNorm statistics of the last two blocks) | seconds |

### TokenFactory outputs (computed once, bundled with the code)

- 5 visual descriptions for each of the 1,000 classes (`data/descriptions_imagenet.json`), JSON mode,
  batches of 10 classes, 6 requests in parallel.
- Visual analogies for the 100 new classes and for 100 seen classes used as validation
  (`data/neighbors_imagenet.json`): from a CLIP shortlist of 20 known classes the LLM picks 1–3 lookalikes
  with weights, e.g. koala → teddy 0.5, wombat 0.3, wallaby 0.2. 25 validation answers failed on the first
  pass and were re-requested.
- Earlier problems with the LLM: it sometimes answered with Arabic class names (fixed in the prompt),
  malformed JSON (fixed with JSON mode and a retry) and timeouts on large batches (fixed with smaller
  batches).

### Choosing the blend of ridge prior and analogies (100 seen classes treated as new)

| Prior for a new class's row | Cosine to the true row | Retrieval among the 100 |
|---|---|---|
| Ridge map from text | 0.712 | 92% |
| Analogies only | 0.629 | 50% |
| Ridge + analogies (β = 0.3) | **0.732** | 85% |
| Ridge + CLIP neighbours (β = 0.2) | 0.729 | 86% |

The blend improves the cosine but lowers retrieval, so it was not clear in advance whether it would help
accuracy; the evaluation below decides.

### Generator

49.1M parameters, 6,000 steps in 29 minutes (3.5 steps/s), loss 0.241 → 0.170.

### Results (100 test tasks per split, 10 classes each)

| | Seen | Half new | All new | All new, hard |
|---|---|---|---|---|
| **T2V-1K** | **70.0%** | **64.1%** | **63.6%** | **47.5%** |
| Prior: TokenFactory analogies | — | 64.1% | 63.6% | 47.5% |
| Prior: CLIP neighbours | — | 64.2% | 62.9% | 48.1% |
| Prior: ridge only | — | 64.0% | 61.2% | 48.3% |
| No known-class gate | — | 64.3% | 60.2% | 44.1% |
| Oracle (fine-tuned) | 72.4% | 74.9% | 74.1% | 64.8% |
| Classifier selection | 69.3% | — | — | — |

0.079 s per generated ResNet-20.

- The analogies help on random new classes (+2.4 over ridge) and not on the hard split (−0.8), where the
  closest relatives were held out with the class. `notebooks/t2v_colab_paired_stats.py` computes paired
  bootstrap intervals for these differences.
- The gate adds 4.2 points on the hard split.

### Diagnostics (50 tasks per split, `results/t2v_1k/diagnose_t2v_1k.json`)

| Probe | Seen | All new | Hard |
|---|---|---|---|
| Right class names / wrong class names | 69.1% / 9.3% | 59.1% / 10.0% | 42.6% / 9.5% |
| Class order reversed | 69.3% | 59.2% | 43.1% |
| Generated body + fine-tuned rows | 70.8% | 71.6% | 59.9% |
| Fine-tuned body + generated rows | 69.8% | 61.4% | 44.4% |
| Zoo-mean body + generated rows | 69.3% | 59.3% | 42.9% |
| Body error: generated / zoo mean | 0.495 / 0.494 | 0.488 / 0.484 | 0.507 / 0.504 |
| Generated rows closer to a training row than to the right one | 96% | 77% | 63% |

- **The generator follows the text** (wrong names → chance) and is equivariant to class order.
- **The rows are the bottleneck.** Fine-tuned rows add 12.5 points on new classes; a fine-tuned body adds about
  0.5 to 2.
- **Negative result: the body is still the zoo's mean.** The zoo now carries task signal (SNR 26.8), but
  the generated body is no closer to the right one than the mean model, and even the right body would add
  little accuracy.
- Snapping of new-class rows to known rows fell from 99.9% (CIFAR) to 77% (ImageNet, random new classes).
- On random new classes, recall is higher for classes far from every seen class (64.5%) than for classes
  close to one (53.7%). Close ones are often fine-grained relatives (dog breeds, for example), which are
  hard to tell apart at 32×32.

## Agent steps measured after T2V-1K (16 October 2026)

Two candidate agent steps were A/B-tested on real ImageNet validation photos (50 per class), same tasks and
same generator for both arms, with the live TokenFactory agent (`experiments/`, logs in `results/agent_ab/`).

| Step | Tasks | Without | With | Difference | Shipped |
|---|---|---|---|---|---|
| **Grounding**: map the user's words onto the vocabulary (requests written with WordNet synonyms, e.g. "Rana catesbeiana", "puma") | 20 × 10 classes | 56.4% | 57.3% | +1.0 ± 0.7 (12 wins, 6 losses) | yes |
| **Contrastive descriptions** for the most similar new-class pairs in a request | 30 × 10 never-seen classes | 58.3% | 58.4% | +0.03 ± 0.07 (11 wins, 11 losses) | no |

Grounding mapped 121 of the 200 task names to exactly the right class; most misses were near-synonym breeds or
species (Russian wolfhound → Irish wolfhound instead of borzoi), and a wrong map can cost accuracy, which keeps
the average gain small on these deliberately rare names. Everyday names map cleanly ("orcas" → killer whale,
"sneakers" → running shoe, "kitties" → tabby; "dogs" correctly stays a new class). The contrastive lines are
averaged with five generic descriptions and the prompt templates, so they barely move the condition.

## What did not work, in one place

- Contrastive agent descriptions for look-alike classes in a request (+0.03 ± 0.07 points; not shipped).
- ε-prediction (diverged; replaced by x₀-prediction).
- Per-tensor weight normalisation for ResNet-20 (chance accuracy; replaced by element normalisation).
- Uniform token sampling (identical classifier rows; replaced by head weighting).
- A larger zoo built from the same classes (no gain on new classes).
- A lighter joint transformer (worse on half-unseen tasks).
- More sampling steps (slower and less accurate than 5).
- Learning a task-specific body: no gain on CIFAR (no signal in the zoo) or on ImageNet (signal present,
  but the generator did not use it, and it is worth little).
- TokenFactory analogies on whole held-out groups (no gain; they help only when relatives were seen).
