"""Report full-repo download progress against the HF file list."""
import json, os, sys, urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
API = f"https://hf-mirror.com/api/models/{REPO}?blobs=true"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "weights")

with urllib.request.urlopen(API, timeout=60) as r:
    data = json.load(r)
files = {s["rfilename"]: s.get("size", 0) for s in data["siblings"] if s["rfilename"] != ".gitattributes"}
total = sum(files.values())
have = 0
done = 0
for name, size in files.items():
    p = os.path.join(DEST, name)
    if os.path.exists(p):
        n = os.path.getsize(p)
        if size == 0 or n == size:
            done += 1
        have += min(n, size) if size else n
print(f"files done: {done}/{len(files)}")
print(f"bytes: {have/1e9:.1f} / {total/1e9:.1f} GB ({100*have/total:.2f}%)")
# per-shard status for the backbone
for name in sorted(n for n in files if n.endswith(".safetensors")):
    size = files[name]
    n = os.path.getsize(os.path.join(DEST, name)) if os.path.exists(os.path.join(DEST, name)) else 0
    state = "OK" if n == size else (f"{n/1e9:.2f}/{size/1e9:.2f} GB" if n else "pending")
    if state != "OK":
        print(f"  {name}: {state}")
