"""Download selected HF safetensors shards with segment resume."""
import subprocess, os, sys, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE = f"https://hf-mirror.com/{REPO}/resolve/main/"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "weights")
SEG = 1024**3
THREADS = 8

def get_len(fn):
    out = subprocess.run(["curl.exe", "-sIL", "--max-time", "60", BASE + fn],
                         capture_output=True, text=True).stdout
    lens = [int(l.split(":")[1].strip()) for l in out.splitlines() if l.lower().startswith("content-length:")]
    return lens[-1] if lens else None

def dl_seg(url, lo, hi, path):
    want = hi - lo + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return True
    for _ in range(5):
        try:
            subprocess.run(["curl.exe", "-sL", "--fail", "--retry", "5", "--retry-delay", "5",
                            "--max-time", "7200", "-r", f"{lo}-{hi}", "-o", path, url], timeout=7300)
        except Exception:
            pass
        if os.path.exists(path) and os.path.getsize(path) == want:
            return True
        time.sleep(5)
    return False

def merge(parts, out_path, total):
    if os.path.exists(out_path) and os.path.getsize(out_path) == total:
        return
    with open(out_path, "wb") as fo:
        for p in parts:
            with open(p, "rb") as fi:
                while True:
                    b = fi.read(64 * 1024 * 1024)
                    if not b:
                        break
                    fo.write(b)

def download_file(fn):
    total = get_len(fn)
    if total is None:
        print(f"WARN: no content-length for {fn}", flush=True)
        return False
    print(f"== {fn} {total/1e9:.3f} GB", flush=True)
    nseg = math.ceil(total / SEG)
    url = BASE + fn
    safe = fn.replace("/", "_")
    parts = [os.path.join(DEST, f"{safe}.part{i:03d}") for i in range(nseg)]
    ok = True
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = {}
        for i in range(nseg):
            lo = i * SEG
            hi = min(lo + SEG, total) - 1
            futs[ex.submit(dl_seg, url, lo, hi, parts[i])] = i
        for f in as_completed(futs):
            if not f.result():
                ok = False
                print(f"FAIL seg {futs[f]} of {fn}", flush=True)
    if not ok:
        return False
    out = os.path.join(DEST, fn)
    merge(parts, out, total)
    good = os.path.getsize(out) == total
    if good:
        for p in parts:
            os.remove(p)
    print(f"{'OK' if good else 'BAD'} {fn} {os.path.getsize(out)/1e9:.3f} GB", flush=True)
    return good

def main():
    os.makedirs(DEST, exist_ok=True)
    files = sys.argv[1:] or [
        "model-00002-of-00048.safetensors",
        "model-00043-of-00048.safetensors",
        "model-00003-of-00048.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    ]
    results = {f: download_file(f) for f in files}
    print("\n== RESULT ==")
    for f, ok in results.items():
        print(("OK  " if ok else "FAIL") + " " + f)
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
