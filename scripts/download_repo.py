"""Download the complete HF checkpoint repository.

Strategy: all 48 safetensors shards first with limited file-level concurrency
(per-file resume via curl -C -); small text files afterwards one by one, since
concurrent small requests get rate-limited by the mirror.
"""
import json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE = f"https://hf-mirror.com/{REPO}/resolve/main/"
API = f"https://hf-mirror.com/api/models/{REPO}?blobs=true"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "weights")
LOG = os.path.join(DEST, "download_repo.log")
WORKERS = 10
SMALL = 20 * 1024 * 1024

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def list_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        data = json.load(r)
    out = []
    for s in data["siblings"]:
        name = s["rfilename"]
        if name in (".gitattributes",):
            continue
        out.append((name, s.get("size", 0)))
    return out

def get_size(name: str) -> int:
    out = subprocess.run(["curl.exe", "-sIL", "--max-time", "60", BASE + name],
                         capture_output=True, text=True).stdout
    lens = [int(l.split(":")[1].strip()) for l in out.splitlines() if l.lower().startswith("content-length:")]
    return lens[-1] if lens else 0

def download_one(name: str, size: int) -> tuple[str, str]:
    path = os.path.join(DEST, name)
    if os.path.exists(path) and (size == 0 or os.path.getsize(path) == size):
        return name, "skip"
    if size == 0:
        size = get_size(name)
    url = BASE + name
    for attempt in range(1, 7):
        subprocess.run(["curl.exe", "-sL", "--fail", "--retry", "5", "--retry-delay", "5",
                        "--max-time", "21600", "-C", "-", "-o", path, url], timeout=21700)
        if os.path.exists(path) and (size == 0 or os.path.getsize(path) == size):
            return name, "done"
        log(f"RETRY {name} attempt={attempt} size={os.path.getsize(path) if os.path.exists(path) else 0}/{size}")
        time.sleep(min(30 * attempt, 120))
    return name, "fail"

def run(files, workers, tag):
    if not files:
        return []
    log(f"START {tag}: {len(files)} files, {sum(s for _, s in files)/1e9:.1f} GB, workers={workers}")
    done = 0
    fails = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, n, s): (n, s) for n, s in files}
        for f in futures:
            n, st = f.result()
            done += 1
            if st == "fail":
                fails.append(n)
            if st != "skip":
                log(f"[{done}/{len(files)}] {st} {n}")
    return fails

def main():
    os.makedirs(DEST, exist_ok=True)
    files = list_files()
    shards = [(n, s) for n, s in files if n.endswith(".safetensors") and s >= SMALL]
    small = [(n, s) for n, s in files if not (n.endswith(".safetensors") and s >= SMALL)]
    import re

    def priority(n: str) -> int:
        m = re.search(r"model-(\d+)-of", n)
        if not m:
            return 9
        idx = int(m.group(1))
        if idx == 3:  # layer 0: verify the forward first
            return 0
        if 4 <= idx <= 10:  # first layers: multi-layer checks
            return 1
        if 11 <= idx <= 42:  # rest of the backbone
            return 2
        if 44 <= idx <= 46:  # MTP heads
            return 3
        if idx in (47, 48):  # engram tables (huge, row-read only)
            return 4
        return 5  # vision / other

    shards.sort(key=lambda x: (priority(x[0]), x[0]))
    fails = run(shards, WORKERS, "shards")
    fails += run(sorted(small), 1, "small-files")
    if fails:
        log(f"FAILED: {fails}")
        return 2
    log("ALL FILES COMPLETE")

if __name__ == "__main__":
    sys.exit(main())
