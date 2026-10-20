"""tinyinfer P1: probe checkpoint layout without downloading full weights.

Reads model.safetensors.index.json + safetensors headers via HTTP Range,
aggregates bytes per component class, and writes report.md + stats.json.
"""
import os, json, subprocess, sys, collections, math
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE = f"https://hf-mirror.com/{REPO}/resolve/main/"
CACHE = r"E:\Develop\Deepseek\tinyinfer\cache"
os.makedirs(CACHE, exist_ok=True)

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F4": 0.5, "FP4": 0.5,
}

def curl_range(path, lo, hi, out):
    url = BASE + path
    for attempt in range(3):
        r = subprocess.run(["curl.exe", "-sL", "--fail", "--retry", "3", "--max-time", "300",
                            "-r", f"{lo}-{hi}", "-o", out, url], capture_output=True)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) == hi - lo + 1:
            return True
    return False

def get_index():
    p = os.path.join(CACHE, "index.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 100000):
        subprocess.run(["curl.exe", "-sL", "--fail", "--retry", "5", "--max-time", "300",
                        "-o", p, BASE + "model.safetensors.index.json"], check=True)
    with open(p, encoding="utf-8") as f:
        return json.load(f)["weight_map"]

def get_header(shard, i):
    tmp8 = os.path.join(CACHE, f"_h8_{i}.bin")
    hdrp = os.path.join(CACHE, f"_hdr_{i}.bin")
    if not curl_range(shard, 0, 7, tmp8):
        return shard, None
    n = int.from_bytes(open(tmp8, "rb").read(), "little")
    if n <= 0 or n > 200 * 1024 * 1024:
        return shard, None
    if not curl_range(shard, 8, 8 + n - 1, hdrp):
        return shard, None
    with open(hdrp, "rb") as f:
        raw = f.read()
    try:
        return shard, json.loads(raw)
    except Exception:
        return shard, None

def numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n

def classify(name):
    if name.startswith("visual."):
        return "vision"
    if name.endswith("engram.embed.weight"):
        return "engram_table"
    if ".experts." in name and "shared_experts" not in name:
        return "routed_expert"
    if ".experts." in name and "shared_experts" in name:
        return "shared_expert"
    return "other"

def main():
    weight_map = get_index()
    shards = sorted(set(weight_map.values()))
    print(f"index: {len(weight_map)} tensors in {len(shards)} shards")

    headers = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(get_header, s, i): s for i, s in enumerate(shards)}
        for f in as_completed(futs):
            s, h = f.result()
            if h is None:
                print(f"WARN: failed header {s}")
            else:
                headers[s] = h
    print(f"headers loaded: {len(headers)}/{len(shards)}")

    stats = collections.defaultdict(lambda: {"bytes": 0, "tensors": 0})
    per_layer = collections.defaultdict(lambda: collections.defaultdict(int))
    examples = collections.defaultdict(list)
    total = 0
    seen = set()
    for shard, hdr in headers.items():
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            seen.add(name)
            dtype = meta.get("dtype", "?")
            shape = meta.get("shape", [])
            n = numel(shape)
            nb = n * DTYPE_BYTES.get(dtype, 0)
            cat = classify(name)
            stats[cat]["bytes"] += nb
            stats[cat]["tensors"] += 1
            total += nb
            if len(examples[cat]) < 5:
                examples[cat].append((name, dtype, shape))
            parts = name.split(".")
            layer = None
            for p in parts:
                if p.isdigit():
                    layer = int(p)
                    break
            if layer is not None and layer < 60:
                per_layer[layer][cat] += nb

    print(f"\nTOTAL: {total/1e9:.1f} GB")
    for cat, st in sorted(stats.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"{cat:14s} {st['bytes']/1e9:8.1f} GB  {st['tensors']:6d} tensors")
        for nm, dt, sh in examples[cat]:
            print(f"    e.g. {nm} {dt} {sh}")

    print("\nPer-layer (GB): layer: attention-ish other / routed experts")
    for layer in sorted(per_layer):
        d = per_layer[layer]
        if d.get("routed_expert", 0) or layer < 3:
            print(f"  L{layer:02d}: other={(d.get('other',0)+d.get('shared_expert',0)+d.get('engram_table',0))/1e9:6.2f} experts={d.get('routed_expert',0)/1e9:6.2f}")

    out = {
        "total_gb": total / 1e9,
        "by_class": {k: {"gb": v["bytes"]/1e9, "tensors": v["tensors"]} for k, v in stats.items()},
        "per_layer_gb": {str(k): {kk: vv/1e9 for kk, vv in v.items()} for k, v in per_layer.items()},
        "missing_from_headers": sorted(set(weight_map) - seen)[:20],
        "extra_in_headers": sorted(seen - set(weight_map))[:20],
    }
    with open(os.path.join(CACHE, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = ["# tinyinfer checkpoint probe", "", f"Total: {total/1e9:.1f} GB", ""]
    lines.append("| class | GB | tensors |")
    lines.append("|---|---|---|")
    for cat, st in sorted(stats.items(), key=lambda kv: -kv[1]["bytes"]):
        lines.append(f"| {cat} | {st['bytes']/1e9:.2f} | {st['tensors']} |")
    with open(os.path.join(CACHE, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nwrote cache/stats.json + cache/report.md")
    return 0

if __name__ == "__main__":
    sys.exit(main())
