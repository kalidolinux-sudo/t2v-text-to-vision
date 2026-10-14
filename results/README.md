# Results

`t2v_1k/` — the ImageNet-1k run of `notebooks/t2v_colab_final_imagenet.py` (Colab T4, 72 min):

- `colab_run_log.txt` — console output (condensed): zoo, generator training, the five evaluations, diagnostics
- `diagnose_t2v_1k.json` — the eight inference-only probes on 50 tasks per split
- `zoo_snr.json` — task signal vs seed noise in the zoo's network bodies

The trained generator is in the [T2V-1K release](https://github.com/kalidolinux-sudo/t2v-text-to-vision/releases/tag/1.0).
