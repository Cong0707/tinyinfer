import json, collections, re
idx = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\index.json", encoding="utf-8"))["weight_map"]
shard_stats = collections.defaultdict(lambda: {"tensors": 0, "layers": set(), "kinds": collections.Counter()})
for name, shard in idx.items():
    s = shard_stats[shard]
    s["tensors"] += 1
    m = re.search(r"layers\.(\d+)\.", name)
    if m: s["layers"].add(int(m.group(1)))
    if name.startswith("visual."): s["kinds"]["vision"] += 1
    elif "engram.embed.weight" in name: s["kinds"]["engram_table"] += 1
    elif ".experts." in name and "shared_experts" not in name: s["kinds"]["expert"] += 1
    elif "embed" in name or "head" in name: s["kinds"]["embed/head"] += 1
    else: s["kinds"]["other"] += 1
for shard in sorted(shard_stats):
    s = shard_stats[shard]
    layers = sorted(s["layers"])
    lr = f"L{layers[0]}-L{layers[-1]}" if layers else "-"
    print(f"{shard}  tensors={s['tensors']:5d} layers={lr:10s} kinds={dict(s['kinds'])}")
