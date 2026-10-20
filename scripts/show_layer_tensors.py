import json, os, sys
CACHE = r"E:\Develop\Deepseek\tinyinfer\cache"
i = int(sys.argv[1]) if len(sys.argv) > 1 else 2
h = json.load(open(os.path.join(CACHE, f"_hdr_{i}.bin"), encoding="utf-8"))
names = [n for n in h if n != "__metadata__"]
others = [n for n in names if ".experts." not in n or "shared_experts" in n]
print(f"shard {i}: {len(names)} tensors, {len(others)} non-routed")
for n in others:
    m = h[n]
    print(f"  {n:60s} {m.get('dtype'):10s} {m.get('shape')}")
print("--- expert example names ---")
for n in [x for x in names if ".experts." in x and "shared_experts" not in x][:6]:
    print("  ", n, h[n].get("dtype"), h[n].get("shape"))
