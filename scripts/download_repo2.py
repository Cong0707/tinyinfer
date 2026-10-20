"""Complete HF repo download with segment-level concurrency across files.

Every large shard is split into 2GiB segments; a global pool of workers pulls
segments (per-segment resume via part files), then completed files are merged
into place and verified. Priority: layer 0 first, then layers 1-8, the rest of
the backbone, MTP heads, engram tables, and finally vision/small files.
"""
import json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE = f"https://hf-mirror.com/{REPO}/resolve/main/"
API = f"https://hf-mirror.com/api/models/{REPO}?blobs=true"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "weights")
LOG = os.path.join(DEST, "download_repo.log")
SEG = 2 * 1024**3
WORKERS = 4
SMALL = 20 * 1024 * 1024

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def list_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        data = json.load(r)
    return [(s["rfilename"], s.get("size", 0)) for s in data["siblings"] if s["rfilename"] != ".gitattributes"]

def priority(name: str) -> int:
    m = re.search(r"model-(\d+)-of", name)
    if not m:
        return 8
    idx = int(m.group(1))
    if idx == 3:
        return 0
    if 4 <= idx <= 12:
        return 1
    if 13 <= idx <= 42:
        return 2
    if 44 <= idx <= 46:
        return 3
    if idx in (47, 48):
        return 4
    return 5

def seg_path(name: str, i: int) -> str:
    return os.path.join(DEST, f"{name}.seg{i:03d}")

def download_seg(url, lo, hi, path):
    want = hi - lo + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return True
    for attempt in range(1, 6):
        subprocess.run(["curl.exe", "-sL", "--fail", "--retry", "5", "--retry-delay", "5",
                        "--max-time", "14400", "-r", f"{lo}-{hi}", "-o", path, url], timeout=14500)
        if os.path.exists(path) and os.path.getsize(path) == want:
            return True
        log(f"RETRY seg {os.path.basename(path)} attempt={attempt} size={os.path.getsize(path) if os.path.exists(path) else 0}/{want}")
        time.sleep(10)
    return False

def merge_file(name: str, total: int, nseg: int) -> bool:
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) == total:
        for i in range(nseg):
            p = seg_path(name, i)
            if os.path.exists(p):
                os.remove(p)
        return True
    tmp = out + ".merging"
    with open(tmp, "wb") as fo:
        for i in range(nseg):
            p = seg_path(name, i)
            if not os.path.exists(p):
                log(f"MERGE MISSING {name} seg {i}")
                return False
            with open(p, "rb") as fi:
                while True:
                    b = fi.read(64 * 1024 * 1024)
                    if not b:
                        break
                    fo.write(b)
    os.replace(tmp, out)
    if os.path.getsize(out) != total:
        log(f"MERGE BAD SIZE {name} {os.path.getsize(out)}/{total}")
        return False
    for i in range(nseg):
        os.remove(seg_path(name, i))
    return True

def main():
    os.makedirs(DEST, exist_ok=True)
    files = list_files()
    big = [(n, s) for n, s in files if n.endswith(".safetensors") and s >= SMALL]
    small = [(n, s) for n, s in files if not (n.endswith(".safetensors") and s >= SMALL)]
    big.sort(key=lambda x: (priority(x[0]), x[0]))
    total_gb = sum(s for _, s in files) / 1e9
    log(f"START {len(files)} files / {total_gb:.1f} GB, seg={SEG//2**20}MiB, workers={WORKERS}")

    # tasks: (priority, file_rank, name, total, seg_idx, lo, hi)
    tasks = []
    for rank, (name, size) in enumerate(big):
        nseg = max(1, (size + SEG - 1) // SEG)
        if os.path.exists(os.path.join(DEST, name)) and os.path.getsize(os.path.join(DEST, name)) == size:
            continue
        for i in range(nseg):
            lo = i * SEG
            hi = min(lo + SEG, size) - 1
            tasks.append((priority(name), rank, name, size, i, lo, hi))
    tasks.sort(key=lambda t: (t[0], t[1], t[4]))
    log(f"segments queued: {len(tasks)}")

    seg_ok = {}
    fails = []

    def do(task):
        _, _, name, size, i, lo, hi = task
        return task, download_seg(BASE + name, lo, hi, seg_path(name, i))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(do, t) for t in tasks]
        for f in futures:
            task, ok = f.result()
            _, _, name, size, i, lo, hi = task
            seg_ok[(name, i)] = ok
            if not ok:
                fails.append(f"{name}#{i}")

    by_file = {}
    for _, _, name, size, i, lo, hi in tasks:
        by_file.setdefault(name, []).append(i)
    for name, size in big:
        if name not in by_file:
            continue
        nseg = max(1, (size + SEG - 1) // SEG)
        if all(seg_ok.get((name, i), False) for i in range(nseg)):
            log(f"MERGE {name} ({size/1e9:.2f} GB)")
            if not merge_file(name, size, nseg):
                fails.append(name)

    if small:
        log(f"SMALL FILES: {len(small)}")
        for name, size in sorted(small):
            out = os.path.join(DEST, name)
            if os.path.exists(out) and (size == 0 or os.path.getsize(out) == size):
                continue
            ok = False
            for attempt in range(1, 6):
                subprocess.run(["curl.exe", "-sL", "--fail", "--retry", "3", "--retry-delay", "5",
                                "--max-time", "600", "-o", out, BASE + name], timeout=700)
                if os.path.exists(out) and (size == 0 or os.path.getsize(out) == size):
                    ok = True
                    break
                time.sleep(10)
            if not ok:
                fails.append(name)
            else:
                log(f"SMALL OK {name}")

    if fails:
        log(f"FAILED: {fails}")
        return 2
    log("ALL FILES COMPLETE")

if __name__ == "__main__":
    sys.exit(main())
