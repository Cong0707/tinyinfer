"""Benchmark tinyinfer.loading.ShardReader on real checkpoint shards."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tinyinfer.loading import ShardIndex, ShardReader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[3] / "cache" / "weights"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    index = ShardIndex(args.model_dir)
    shard = index.weight_map[f"layers.{args.layer}.ffn.experts.0.w1.weight"]
    size = (index.model_dir / shard).stat().st_size
    readable = []
    for e in range(384):
        blob = index.expert_blob(args.layer, e)
        if all(seg.file_offset + seg.nbytes <= size for seg in blob.segments):
            readable.append(e)
        if len(readable) >= args.experts:
            break
    if not readable:
        print(f"no complete expert blob inside {shard} ({size/1e9:.2f} GB)")
        return 1
    print(f"shard={shard} size={size/1e9:.2f} GB readable experts={readable}")
    blob = index.expert_blob(args.layer, readable[0])
    print(f"blob={blob.total_bytes} B segments={[(s.nbytes, s.file_offset % 4096) for s in blob.segments]}")

    for direct in (False, True):
        reader = ShardReader(index, threads=args.threads, direct=direct)
        for mode in ("single", "parallel"):
            times = []
            total = 0
            for _ in range(args.repeats):
                t = time.perf_counter()
                if mode == "single":
                    bufs = [reader.read_blob(index.expert_blob(args.layer, e)) for e in readable]
                else:
                    bufs = reader.read_many([index.expert_blob(args.layer, e) for e in readable])
                times.append(time.perf_counter() - t)
                total = sum(len(b) for b in bufs)
            best = min(times)
            print(f"direct={str(direct):<5s} {mode:<8s} {total/1e6:7.1f} MB best={best*1e3:7.2f} ms "
                  f"{total/best/1e9:5.2f} GB/s")
        print("  reader stats:", {k: round(v, 4) for k, v in reader.stats().items()})
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
