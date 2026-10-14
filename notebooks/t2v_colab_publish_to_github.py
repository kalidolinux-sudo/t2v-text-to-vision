# T2V · نقل نتائج T2V-1K من Google Drive إلى GitHub (خلية واحدة، دقائق، بدون GPU)
#   - النتائج الصغيرة (التقييم، التشخيص، meta، SNR، النموذج العام، النموذج التجريبي) → results/t2v_1k/ في المستودع
#   - المولّد t2v_1k.pt (198 MB) → مرفق في Release v1.0
#   - اختياري: حديقة النماذج (~0.8 GB) → مرفقات في نفس الـRelease
# يطلب توكن GitHub عند التشغيل (لا يُكتب في الخلية ولا يُطبع ولا يُحفظ):
#   github.com/settings/personal-access-tokens/new → Only select repositories: t2v-text-to-vision
#   → Repository permissions: Contents = Read and write → Generate token. احذفه بعد الاستخدام.
REPO = "kalidolinux-sudo/t2v-text-to-vision"
OUT = "/content/drive/MyDrive/t2v_imagenet"
TAG = "v1.0"
UPLOAD_ZOO = False          # True = ارفع أيضاً ملفات الحديقة كمرفقات (~0.8 GB)

import getpass, glob, json, os, shutil, subprocess

import requests

if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
assert os.path.exists(f"{OUT}/t2v_1k.pt"), f"لم أجد {OUT}/t2v_1k.pt"

token = getpass.getpass("GitHub token (Contents: read & write on this repo only): ").strip()
API, H = "https://api.github.com", {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
r = requests.get(f"{API}/repos/{REPO}", headers=H)
assert r.ok, f"التوكن لا يصل للمستودع ({r.status_code}). تأكد من الصلاحية Contents: Read and write"


def git(*args, cwd=None):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((p.stdout + p.stderr).replace(token, "***"))
    return p.stdout


# ---------------------------------------------------------------- 1. small results into the repository
work = "/content/t2v_repo"
shutil.rmtree(work, ignore_errors=True)
git("clone", "--depth", "1", f"https://x-access-token:{token}@github.com/{REPO}.git", work)
dst = f"{work}/results/t2v_1k"
os.makedirs(dst, exist_ok=True)
copied = []
for pattern in ("eval_*.json", "diagnose_*.json", "zoo/meta.json", "zoo/zoo_snr.json", "zoo/generic.pt",
                "zoo/zoo_*.log", "zoo_*.log", "demo/*"):
    for f in glob.glob(f"{OUT}/{pattern}"):
        if os.path.isfile(f) and os.path.getsize(f) < 50e6:
            sub = "demo/" if "/demo/" in f else ""
            os.makedirs(f"{dst}/{sub}", exist_ok=True)
            shutil.copy(f, f"{dst}/{sub}{os.path.basename(f)}")
            copied.append(sub + os.path.basename(f))
git("add", "-f", "results", cwd=work)
if git("status", "--porcelain", cwd=work).strip():
    git("-c", "user.name=T2V", "-c", "user.email=t2v@users.noreply.github.com", "commit", "-q", "-m",
        "T2V-1K results from the Colab run (evaluation, diagnostics, zoo meta and SNR, generic model, demo)", cwd=work)
    git("push", "-q", "origin", "HEAD:main", cwd=work)
print(f"1) pushed {len(copied)} files to results/t2v_1k/: {', '.join(sorted(copied))}")

# ---------------------------------------------------------------- 2. release with the generator
r = requests.get(f"{API}/repos/{REPO}/releases/tags/{TAG}", headers=H)
if r.status_code == 404:
    body = ("Trained T2V generators.\n\n- `t2v_1k.pt`: ImageNet-1k 32x32, 1000-class vocabulary (900 seen, 100 held out). "
            "Seen 70.0%, all-new 63.6%, all-new hard 47.5% (oracle 72.4 / 74.1 / 64.8), 0.079 s per ResNet-20.\n"
            "- `t2v_tinacnn_cifar100.pt` is in `checkpoints/` in the repository.\n\n"
            "Use: put the file in `checkpoints/` and run `python app.py --ckpt checkpoints/t2v_1k.pt`.")
    r = requests.post(f"{API}/repos/{REPO}/releases", headers=H,
                      json={"tag_name": TAG, "target_commitish": "main", "name": "T2V-1K", "body": body})
r.raise_for_status()
rel = r.json()
have = {a["name"] for a in rel.get("assets", [])}
files = [f"{OUT}/t2v_1k.pt"] + (sorted(glob.glob(f"{OUT}/zoo/*_shard*.pt")) if UPLOAD_ZOO else [])
for f in files:
    name = ("zoo_" if "/zoo/" in f else "") + os.path.basename(f)
    if name in have:
        print(f"   {name}: already attached")
        continue
    size = os.path.getsize(f)
    print(f"   uploading {name} ({size / 1e6:.0f} MB) ...", flush=True)
    with open(f, "rb") as fh:
        up = requests.post(f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets",
                           params={"name": name}, data=fh,
                           headers={**H, "Content-Type": "application/octet-stream", "Content-Length": str(size)})
    up.raise_for_status()
print(f"2) release: {rel['html_url']}")
print(f"   download link: https://github.com/{REPO}/releases/download/{TAG}/t2v_1k.pt")
del token, H
print("\nتم. احذف التوكن من github.com/settings/personal-access-tokens بعد الانتهاء.")
