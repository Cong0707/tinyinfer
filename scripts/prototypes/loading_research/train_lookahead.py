"""Fit and evaluate a CPU lookahead router predictor from a router trace.

For each target layer t, ridge-regress gate logits(t) from the hidden state of
an earlier layer t - lead.  We report top-k precision/recall of the predicted
expert set against the real top-k routes: that is exactly the number the
prefetcher's break-even model consumes (see docs/loading.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from tinyinfer.loading import LookaheadRouterPredictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()

    trace = np.load(args.trace)
    layers = sorted(int(k.split("_l")[1]) for k in trace.files if k.startswith("logits_l"))
    weights, bias = {}, {}
    rows = []
    for target in layers:
        source = target - args.lead
        if source < 0:
            continue
        hkey, ykey, rkey = f"hidden_l{source}", f"logits_l{target}", f"routes_l{target}"
        if not all(k in trace.files for k in (hkey, ykey, rkey)):
            continue
        X, Y, R = trace[hkey], trace[ykey], trace[rkey]
        n = min(len(X), len(Y), len(R))
        X, Y, R = X[:n].astype(np.float32), Y[:n].astype(np.float32), R[:n]
        # simple chronological split: last 20% held out
        cut = max(1, int(n * 0.8))
        W, b = LookaheadRouterPredictor.fit(list(zip(X[:cut], Y[:cut])), ridge=args.ridge)
        weights[target], bias[target] = W, b
        logits = X[cut:] @ W.T + b
        top = np.argsort(-logits, axis=1)[:, : args.top_k]
        hit = np.array([len(set(top[i]) & set(R[cut + i])) for i in range(len(top))])
        precision = hit.sum() / max(1, top.size)
        recall = hit.sum() / max(1, R[cut:].size)
        rows.append((target, len(R), precision, recall))
    print(f"{'target':>6s} {'samples':>8s} {'precision@k':>12s} {'recall@k':>9s}")
    for target, n, p, r in rows:
        print(f"{target:>6d} {n:>8d} {p:>12.3f} {r:>9.3f}")
    if rows:
        print(f"mean precision={np.mean([r[2] for r in rows]):.3f} "
              f"mean recall={np.mean([r[3] for r in rows]):.3f}")
    if args.out:
        LookaheadRouterPredictor(weights, top_k=args.top_k, bias=bias).save_npz(args.out)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
