"""Expert-cache / prefetch sweep for tinyinfer loading design.

Synthetic routing is Zipf(global) + Dirichlet(topic) + temporal reuse; when a
real router trace (npz from collect_router_trace.py) is available, pass
--trace to replace the synthetic workload.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tinyinfer.loading import (  # noqa: E402
    ModelGeometry,
    format_sim_table,
    simulate_cache,
    simulate_prefetch,
    simulate_workload,
)

BLOB = ModelGeometry().expert_bytes()
LAYERS, EXPERTS, K = 40, 384, 6


def workload_stats(routes) -> dict[str, float]:
    temporal = total = 0
    for a, b in zip(routes, routes[1:]):
        for la, lb in zip(a, b):
            total += 1
            temporal += len(set(la) & set(lb))
    per_token = [set(e for layer in tok for e in layer) for tok in routes]
    flat = [e for layer in routes[-1] for e in layer]
    return {
        "temporal_expert_overlap": temporal / max(1, len(routes) * LAYERS * K),
        "last_token_distinct": len(set(flat)),
    }


def load_trace(path: Path):
    data = np.load(path)
    routes = []
    if "routes" in data:
        arr = data["routes"]  # [tokens, layers, k]
        for tok in arr:
            routes.append([tuple(int(e) for e in layer) for layer in tok])
    return routes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--zipf", type=float, default=0.9)
    ap.add_argument("--temporal", type=float, default=0.25)
    ap.add_argument("--topic-reuse", type=float, default=0.5)
    ap.add_argument("--compute-ms", type=float, default=3.0)
    ap.add_argument("--read-gbs", type=float, default=8.8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.trace:
        routes = load_trace(Path(args.trace))
        source = f"real trace {args.trace}"
    else:
        routes = simulate_workload(steps=args.steps, zipf_s=args.zipf,
                                   temporal_reuse=args.temporal,
                                   topic_reuse=args.topic_reuse, seed=0)
        source = "synthetic Zipf+topic+temporal"
    stats = workload_stats(routes)

    rows = []
    for gb in (4, 8, 12, 16, 20, 24, 28, 32):
        cap = int(gb * 1e9 // BLOB)
        for policy in ("lru", "arc", "s3fifo"):
            row = simulate_cache(routes, cap, policy)
            row.update({"capacity_gb": gb})
            rows.append(row)
        row = simulate_cache(routes, cap, "arc", pin_per_layer=20)
        row["policy"] = "arc+pin20"
        row.update({"capacity_gb": gb})
        rows.append(row)

    lines = [f"workload: {source}, tokens={len(routes)}, layers={LAYERS}, k={K}",
             f"temporal_expert_overlap={stats['temporal_expert_overlap']:.3f}", "",
             format_sim_table(rows), "", "prefetch (k=2, layers serial compute+read):"]
    for precision in (0.4, 0.5, 0.6, 0.7, 0.8):
        row = simulate_prefetch(routes, precision=precision, recall=0.5,
                                k_prefetch=2, compute_ms=args.compute_ms,
                                read_gbs=args.read_gbs)
        lines.append(f"  precision={precision:.1f}  speedup={row['speedup']:.3f}  "
                     f"wasted={row['wasted_gb']/len(routes):.3f} GB/token  "
                     f"total={row['total_gb_per_token']:.2f} GB/token")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
