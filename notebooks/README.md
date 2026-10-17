# Notebooks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kalidolinux-sudo/t2v-text-to-vision/blob/main/notebooks/T2V_demo.ipynb)
**[`T2V_demo.ipynb`](T2V_demo.ipynb)** is the interactive demo: text → generated vision model → accuracy on
real ImageNet photos → the full app on a public link. *Runtime → Run all*; a few minutes, GPU optional.

Each `.py` file is one Google Colab cell: paste it into a cell and run it. The cells bundle the code they
need, so they work without cloning the repository, and none of them contains an API key.

| Cell | What it does | Time (T4) |
|---|---|---|
| [`t2v_colab_final_imagenet.py`](t2v_colab_final_imagenet.py) | **T2V-1K end to end**: ImageNet-1k zoo, generator, evaluation, diagnostics, demo | ~72 min |
| [`t2v_colab_paired_stats.py`](t2v_colab_paired_stats.py) | Paired bootstrap intervals for the prior comparisons of the T2V-1K run | seconds |
| [`t2v_colab_publish_to_github.py`](t2v_colab_publish_to_github.py) | Copies the run's results from Google Drive to this repository and its release | minutes |

[`experiments/`](experiments/) holds the upgrade tests run after T2V-1K, with their results.

[`cifar100_dev/`](cifar100_dev/) holds the cells of the CIFAR-100 development runs described in the
README's development log (ResNet-20 zoo, generator versions v4 to v7, sampling sweep, diagnostics),
plus a Kaggle script.
