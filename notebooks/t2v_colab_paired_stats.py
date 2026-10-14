# T2V-1K · فحص الفروق (ثواني، بدون GPU): هل فرق TokenFactory حقيقي أم ضمن الضجيج؟
# مقارنة مزدوجة على نفس الـ100 مهمة لكل نوع، مع فترة ثقة 95% (bootstrap).
import json, random

OUT = "/content/drive/MyDrive/t2v_imagenet"
import os
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")

R = {t: json.load(open(f"{OUT}/eval_{t}.json")) for t in ("llm", "clip", "ridge", "nogate")}


def per_task(tag, split):
    return [sum(r["generated"]) / len(r["generated"]) for r in R[tag][split]["tasks"]]


def paired(a, b, split, n_boot=10000):
    x, y = per_task(a, split), per_task(b, split)
    assert [r["classes"] for r in R[a][split]["tasks"]] == [r["classes"] for r in R[b][split]["tasks"]]
    d = [p - q for p, q in zip(x, y)]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    wins = sum(v > 0 for v in d), sum(v < 0 for v in d)
    return sum(d) / len(d), lo, hi, wins


print(f"{'comparison':34s} {'split':16s} {'diff':>7s} {'95% CI':>17s}  tasks better/worse")
for a, b, label in (("llm", "ridge", "TokenFactory vs ridge only"), ("llm", "clip", "TokenFactory vs CLIP neighbours"),
                    ("clip", "ridge", "CLIP neighbours vs ridge only"), ("ridge", "nogate", "known-class gate vs none")):
    for split in ("eval_unseen", "eval_novel", "eval_novel_hard"):
        m, lo, hi, (w, l) = paired(a, b, split)
        sig = "  *" if lo > 0 or hi < 0 else ""
        print(f"{label:34s} {split:16s} {100 * m:+6.1f}  [{100 * lo:+5.1f}, {100 * hi:+5.1f}]  {w:3d}/{l:<3d}{sig}")
print("\n* = the 95% interval excludes 0 (the difference is not noise)")
