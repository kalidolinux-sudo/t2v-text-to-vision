# Notebooks

Each `.py` file is one Google Colab cell: paste it into a cell and run it. The cells bundle the code they
need, so they work without cloning the repository, and none of them contains an API key.

| Cell | What it does | Time (T4) |
|---|---|---|
| [`t2v_colab_final_imagenet.py`](t2v_colab_final_imagenet.py) | **T2V-1K end to end**: ImageNet-1k zoo, generator, evaluation, diagnostics, demo | ~72 min |
| [`t2v_colab_paired_stats.py`](t2v_colab_paired_stats.py) | Paired bootstrap intervals for the prior comparisons of the T2V-1K run | seconds |
| [`t2v_colab_publish_to_github.py`](t2v_colab_publish_to_github.py) | Copies the run's results from Google Drive to this repository and its release | minutes |

[`cifar100_dev/`](cifar100_dev/) holds the cells of the CIFAR-100 development runs described in the
README's development log (ResNet-20 zoo, generator versions v4 to v7, sampling sweep, diagnostics),
plus a Kaggle script.
