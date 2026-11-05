#!/usr/bin/env python
"""What the engine will do on a given card, from measurements taken on an 8GB one.

The planner is a solver over bytes and bandwidths, so it can answer for hardware that is
not in this machine -- but only if its inputs are the checkpoint's real sizes and its
prices are measured, not guessed. This script prints, for a list of cards:

  - what the planner decides (resident layers, slots, host cache, MTP on/off)
  - the expert bytes a decode token must move, and the H2D time at that card's link
  - which term dominates, so it is clear whether the card is compute- or IO-bound

Run: python scripts/predict.py --model <dir> [--cards 8,24,32,80,141]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

GB = 1024 ** 3

# Measured on this box (RTX 5060 Laptop, PCIe):
#   pinned H2D 25.5-28.5 GB/s, pageable 12.4-15.3, copy_ dispatch 40-56us
# Datacenter/consumer links scale from the spec sheet instead; the ratio that matters
# here is only "how many bytes must cross the link per token".
LINK_GBPS = {
    8: 26.0,      # PCIe 4.0 x8 (this laptop)
    16: 26.0,
    24: 50.0,     # PCIe 4.0 x16
    32: 50.0,
    48: 50.0,
    80: 50.0,
    141: 50.0,
}


def link_gbps(vram_gb: float) -> float:
    for k in sorted(LINK_GBPS):
        if vram_gb <= k:
            return LINK_GBPS[k]
    return 50.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cards', default='8,24,32,80,141')
    ap.add_argument('--ram', type=float, default=64.0)
    args = ap.parse_args()

    from tinyinfer.loading import ShardIndex
    from tinyinfer.resources import Card, Hardware, model_profile, plan

    prof = model_profile(ShardIndex(args.model))
    topk = int(prof.get('topk') or 0)
    n_layers = int(prof.get('n_layers') or 40)
    expert_b = float(prof['expert_bytes'])
    need_slots = topk * n_layers

    print(f'model : {args.model}')
    print(f'        {n_layers} layers, layer {prof["layer_bytes"]/2**20:.0f}MB, '
          f'expert {expert_b/2**20:.1f}MB, topk {topk}')
    print(f'        a decode token touches {need_slots} expert blobs '
          f'= {need_slots*expert_b/GB:.2f} GB')
    print()
    hdr = (f'{"card":>10s} {"layers":>7s} {"stream":>7s} {"slots":>7s} '
           f'{"mtp":>4s} {"expertGB/tok":>13s} {"H2D ms":>7s} {"bound":>9s}')
    print(hdr)
    print('-' * len(hdr))

    for txt in args.cards.split(','):
        vram = float(txt)
        hw = Hardware(name=f'{vram:.0f}G', vram_gb=vram, ram_gb=args.ram,
                      ram_avail_gb=args.ram * 0.8,
                      cards=[Card(0, f'{vram:.0f}G', vram, vram * 0.95)])
        p = plan(hw=hw, n_layers=n_layers, model=prof, speculation=True)
        # experts not covered by the resident slot pool have to cross the link
        covered = min(p.expert_slots, need_slots)
        miss_frac = 1.0 - covered / need_slots if need_slots else 0.0
        expert_gb = need_slots * expert_b * miss_frac / GB
        # streamed layers cross it too
        layer_gb = p.stream_layers * float(prof['layer_bytes']) / GB
        total_gb = expert_gb + layer_gb
        ms = total_gb / link_gbps(vram) * 1000
        bound = 'compute' if total_gb < 0.5 else ('IO' if ms > 30 else 'mixed')
        print(f'{vram:>10.0f} {p.gpu_layers:>7d} {p.stream_layers:>7d} '
              f'{p.expert_slots:>7d} {str(p.speculation):>4s} '
              f'{expert_gb:>13.2f} {ms:>7.0f} {bound:>9s}')
    print()
    print('expertGB/tok counts only the expert bytes NOT held in the VRAM slot pool.')
    print('A row at 0.00 GB is the interesting one: the 296GB expert table is being')
    print('served without touching the link at all.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
