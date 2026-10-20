"""Complete HF repo download with segment-level concurrency + incremental merge.

Each large shard is split into SEG-byte segments; a global pool of WORKERS pulls
segments (per-segment resume via part files). As soon as ALL segments of a file
are present it is merged into place immediately (so priority files land early),
verified by size, and its segments removed. A single-instance lock prevents
double-downloading. Priority: layer 0 first, then layers 4-12, the rest of the
backbone, MTP heads, engram tables, and finally vision/small files.

Usage:
  python download_repo2.py [--seg-mb 100] [--workers 4] [--only-small]
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE = f"https://hf-mirror.com/{REPO}/resolve/main/"
API = f"https://hf-mirror.com/api/models/{REPO}?blobs=true"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "weights")
LOG = os.path.join(DEST, "download_repo.log")
LOCK = os.path.join(DEST, ".download_repo2.lock")
SEG = 100 * 1024 * 1024
WORKERS = 4
SMALL = 20 * 1024 * 1024

_log_lock = threading.Lock()

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with _log_lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except Exception:
            pass

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
    return os.path.join(DEST, f"{name}.seg{i:04d}")

def seg_complete(path, want):
    try:
        return os.path.exists(path) and os.path.getsize(path) == want
    except OSError:
        return False

def download_seg(url, lo, hi, path, retries=5):
    want = hi - lo + 1
    if seg_complete(path, want):
        return True
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(["curl.exe", "-sS", "-L", "--fail",
                            "--connect-timeout", "30",
                            "--speed-limit", "20000", "--speed-time", "90",
                            "--max-time", "3600",
                            "-r", f"{lo}-{hi}", "-o", path, url],
                           timeout=3700, capture_output=True)
        except subprocess.TimeoutExpired:
            pass
        if seg_complete(path, want):
            return True
        log(f"RETRY seg {os.path.basename(path)} attempt={attempt} size={os.path.getsize(path) if os.path.exists(path) else 0}/{want}")
        time.sleep(min(60, 5 * attempt))
    return False

def seg_bytes():
    total = 0
    try:
        with os.scandir(DEST) as it:
            for e in it:
                try:
                    if ".seg" in e.name and e.is_file():
                        total += e.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total

class Heartbeat(threading.Thread):
    def __init__(self, stop, state):
        super().__init__(daemon=True)
        self.stop_ev = stop
        self.state = state

    def run(self):
        last = seg_bytes()
        last_t = time.time()
        while not self.stop_ev.wait(60):
            now = seg_bytes()
            dt = max(1e-6, time.time() - last_t)
            rate = (now - last) / dt / 1e6
            with _log_lock:
                state["last_rate"] = rate
            log(f"HB segs={now/1e9:.2f}GB rate={rate:.2f}MB/s merged={self.state.get('merged', 0)} fails={self.state.get('fails', 0)}")
            last, last_t = now, time.time()

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
    if os.path.getsize(tmp) != total:
        log(f"MERGE BAD SIZE {name} {os.path.getsize(tmp)}/{total}")
        os.remove(tmp)
        return False
    os.replace(tmp, out)
    for i in range(nseg):
        try:
            os.remove(seg_path(name, i))
        except OSError:
            pass
    return True

def acquire_lock():
    try:
        import msvcrt
        fd = os.open(LOCK, os.O_RDWR | os.O_CREAT)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            os.close(fd)
            return None
        os.write(fd, str(os.getpid()).encode())
        os.lseek(fd, 0, 0)
        return fd
    except Exception as e:
        log(f"LOCK ERROR {e}")
        return -1

def download_small(files):
    fails = []
    for name, size in sorted(files):
        out = os.path.join(DEST, name)
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(out) and (size == 0 or os.path.getsize(out) == size):
            continue
        ok = False
        for attempt in range(1, 6):
            try:
                subprocess.run(["curl.exe", "-sS", "-L", "--fail", "--retry", "3", "--retry-delay", "5",
                                "--connect-timeout", "30", "--max-time", "600", "-o", out, BASE + name],
                               timeout=700, capture_output=True)
            except subprocess.TimeoutExpired:
                pass
            if os.path.exists(out) and (size == 0 or os.path.getsize(out) == size):
                ok = True
                break
            log(f"RETRY small {name} attempt={attempt}")
            time.sleep(10 * attempt)
        if ok:
            log(f"SMALL OK {name} ({size} bytes)")
        else:
            fails.append(name)
    return fails

def main():
    global SEG, WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-mb", type=int, default=SEG // 2**20)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--only-small", action="store_true")
    args = ap.parse_args()
    SEG = args.seg_mb * 1024 * 1024
    WORKERS = args.workers

    os.makedirs(DEST, exist_ok=True)
    lock_fd = acquire_lock()
    if lock_fd is None:
        log("ANOTHER INSTANCE RUNNING - exit")
        return 3

    files = list_files()
    big = [(n, s) for n, s in files if n.endswith(".safetensors") and s >= SMALL]
    small = [(n, s) for n, s in files if not (n.endswith(".safetensors") and s >= SMALL)]
    big.sort(key=lambda x: (priority(x[0]), x[0]))
    total_bytes = sum(s for _, s in files)
    log(f"START {len(files)} files / {total_bytes/1e9:.1f} GB, seg={SEG//2**20}MiB, workers={WORKERS}")

    state = {"merged": 0, "fails": 0, "last_rate": 0.0}
    stop_ev = threading.Event()
    hb = Heartbeat(stop_ev, state)
    hb.start()

    merged_bytes = 0
    for name, size in big:
        p = os.path.join(DEST, name)
        if os.path.exists(p) and os.path.getsize(p) == size:
            merged_bytes += size

    fails = []
    if not args.only_small:
        tasks = []
        for rank, (name, size) in enumerate(big):
            p = os.path.join(DEST, name)
            if os.path.exists(p) and os.path.getsize(p) == size:
                continue
            nseg = max(1, (size + SEG - 1) // SEG)
            for i in range(nseg):
                lo = i * SEG
                hi = min(lo + SEG, size) - 1
                tasks.append((priority(name), rank, name, size, nseg, i, lo, hi))
        tasks.sort(key=lambda t: (t[0], t[1], t[5]))
        log(f"segments queued: {len(tasks)}")

        remaining = {}
        nsegs = {}
        bad = {}
        for t in tasks:
            remaining[t[2]] = remaining.get(t[2], 0) + 1
            nsegs[t[2]] = t[4]
        merge_pool = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=1)
        merge_lock = threading.Lock()
        merge_list = []

        def finalize(name, size, nseg):
            log(f"MERGE start {name} ({size/1e9:.2f} GB, {nseg} segs)")
            if merge_file(name, size, nseg):
                state["merged"] += 1
                log(f"MERGED {name} ({size/1e9:.2f} GB)")
            else:
                fails.append(name)
                state["fails"] += 1

        def do(task):
            _, _, name, size, nseg, i, lo, hi = task
            return task, download_seg(BASE + name, lo, hi, seg_path(name, i))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        state["tasks"] = len(tasks)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(do, t) for t in tasks]
            for f in as_completed(futs):
                task, ok = f.result()
                name, size, nseg = task[2], task[3], task[4]
                with merge_lock:
                    if not ok:
                        bad[name] = True
                    remaining[name] -= 1
                    done = remaining[name] <= 0
                if done:
                    if bad.get(name):
                        log(f"FILE FAILED {name} (missing segments)")
                        fails.append(name)
                        state["fails"] += 1
                    else:
                        merge_pool.submit(finalize, name, size, nseg)

        merge_pool.shutdown(wait=True)
        big_bytes_done = sum(s for n, s in big if os.path.exists(os.path.join(DEST, n)) and os.path.getsize(os.path.join(DEST, n)) == s)
        log(f"BIG DONE {big_bytes_done/1e9:.1f} / {sum(s for _, s in big)/1e9:.1f} GB")

    stop_ev.set()
    hb.join(timeout=2)

    if small:
        got = sum(1 for n, s in small if os.path.exists(os.path.join(DEST, n)) and (s == 0 or os.path.getsize(os.path.join(DEST, n)) == s))
        log(f"SMALL FILES: {len(small)} ({got} already ok)")
        fails += download_small(small)

    if fails:
        log(f"FAILED: {fails}")
        return 2
    log("ALL FILES COMPLETE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
