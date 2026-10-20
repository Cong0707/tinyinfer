import json, os, glob
CACHE = r"E:\Develop\Deepseek\tinyinfer\cache"
for i in [0, 1, 2, 42, 43, 44, 45, 46, 47]:
    p = os.path.join(CACHE, f"_hdr_{i}.bin")
    if not os.path.exists(p):
        continue
    h = json.load(open(p, encoding="utf-8"))
    print(f"--- shard index {i} (model-{i+1:05d}) : {len(h)-1 if '__metadata__' in h else len(h)} tensors")
    for name, meta in list(h.items())[:14]:
        if name == "__metadata__":
            continue
        print(f"    {name} {meta.get('dtype')} {meta.get('shape')}")
