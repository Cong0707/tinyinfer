"""Report full-repo download progress against the HF file list.

Counts finished files by exact size, plus in-flight segment bytes, so progress is
visible while big shards are still being assembled. Also prints the observed rate
since the previous run (state kept in _progress_state.json).
"""
import json, os, sys, time, urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
API = f"https://hf-mirror.com/api/models/{REPO}?blobs=true"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "weights")
STATE = os.path.join(DEST, "_progress_state.json")
CACHE = os.path.join(DEST, "_api_cache.json")


def load_files():
    try:
        if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < 1800:
            with open(CACHE, encoding="utf-8") as f:
                d = json.load(f)
            return d
    except Exception:
        pass
    with urllib.request.urlopen(API, timeout=60) as r:
        data = json.load(r)
    files = {s["rfilename"]: s.get("size", 0) for s in data["siblings"] if s["rfilename"] != ".gitattributes"}
    try:
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(files, f)
    except Exception:
        pass
    return files


def main():
    files = load_files()
    total = sum(files.values())
    seg_bytes = 0
    seg_count = 0
    for n in os.listdir(DEST):
        if ".seg" in n:
            try:
                seg_bytes += os.path.getsize(os.path.join(DEST, n))
                seg_count += 1
            except OSError:
                pass
    have = 0
    done = 0
    partial = []
    for name, size in files.items():
        p = os.path.join(DEST, name)
        if os.path.exists(p):
            n = os.path.getsize(p)
            if size == 0 or n == size:
                done += 1
                have += size
            else:
                have += min(n, size)
                partial.append((name, n, size))
        else:
            partial.append((name, 0, size))
    have += seg_bytes
    have = min(have, total)

    now = time.time()
    rate = None
    prev_n, prev_t = None, None
    try:
        with open(STATE, encoding="utf-8") as f:
            st = json.load(f)
        prev_n, prev_t = st["bytes"], st["t"]
    except Exception:
        pass
    rate = (have - prev_n) / max(1e-6, now - prev_t) if prev_n is not None and now > prev_t else None
    try:
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump({"bytes": have, "t": now}, f)
    except Exception:
        pass

    print(f"files done: {done}/{len(files)}")
    print(f"bytes: {have/1e9:.2f} / {total/1e9:.1f} GB ({100*have/total:.2f}%)")
    print(f"segments on disk: {seg_count} files / {seg_bytes/1e9:.2f} GB")
    if rate is not None:
        rem = total - have
        eta_h = rem / rate / 3600 if rate > 0 else float("inf")
        print(f"rate since last sample: {rate/1e6:.2f} MB/s   ETA at this rate: {eta_h:.1f} h")
    print(f"incomplete files: {len(partial)}")
    for name, n, size in sorted(partial):
        if n == 0:
            print(f"  {name}: pending ({size/1e9:.2f} GB)")
        else:
            print(f"  {name}: {n/1e9:.2f}/{size/1e9:.2f} GB")
    # last heartbeat from the downloader log
    log = os.path.join(DEST, "download_repo.log")
    if os.path.exists(log):
        hbs = []
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if " HB " in line:
                    hbs.append(line.rstrip())
        if hbs:
            print("last heartbeat: " + hbs[-1])


if __name__ == "__main__":
    main()
