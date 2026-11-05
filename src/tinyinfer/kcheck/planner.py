"""Planner invariants across hardware it will never see on this box.

The planner is the only component that decides everything else (resident layers, expert
slots, host cache, whether MTP fits), and it is a solver over measured bytes rather than
a table of card names. These cases pin down the invariants that a solver regression
would break: layer counts stay in range, slot counts never go negative, and a card too
small to hold the streaming skeleton says so instead of returning nonsense.
"""
from __future__ import annotations

from . import Case

GB = 1024 ** 3


def _profile():
    import os

    from ..loading import ShardIndex
    from ..resources import model_profile

    path = os.environ.get('TI_MODEL')
    if not path:
        return None
    return model_profile(ShardIndex(path))


def register(report) -> None:
    prof = _profile()
    if prof is None:
        report.add(Case('planner', 'skipped (set TI_MODEL to a checkpoint dir)', True))
        return

    from ..resources import Card, Hardware, plan

    n_layers = int(prof.get('n_layers') or 40)
    cases = [
        ('2GB too small', 2.0, 8.0, 6.0),
        ('4GB', 4.0, 16.0, 12.0),
        ('8GB', 8.0, 47.0, 30.0),
        ('16GB', 16.0, 64.0, 50.0),
        ('24GB', 24.0, 64.0, 50.0),
        ('1TB no overflow', 1024.0, 2048.0, 1800.0),
    ]
    for name, vram, ram, avail in cases:
        hw = Hardware(name=name, vram_gb=vram, ram_gb=ram, ram_avail_gb=avail,
                      cards=[Card(0, name, vram, vram * 0.95)])
        try:
            p = plan(hw=hw, n_layers=n_layers, model=prof, speculation=True)
        except Exception as exc:
            report.add(Case(f'planner {name}', f'raised {exc!r}', False))
            continue
        in_range = 0 <= p.gpu_layers <= n_layers
        non_negative = p.stream_layers >= 0 and p.expert_slots >= 0 and p.cache_bytes >= 0
        consistent = p.stream_layers == n_layers - p.gpu_layers
        report.add(Case(f'planner {name}',
                        f'layers={p.gpu_layers}/{n_layers} stream={p.stream_layers} '
                        f'slots={p.expert_slots}',
                        in_range and non_negative and consistent))

    # A card that cannot hold the streaming skeleton must say so, not return a plan
    hw = Hardware(name='2GB', vram_gb=2.0, ram_gb=8.0, ram_avail_gb=6.0,
                  cards=[Card(0, '2GB', 2.0, 1.9)])
    p = plan(hw=hw, n_layers=n_layers, model=prof, speculation=False)
    report.add(Case('planner refuses a card below the skeleton',
                    f'fits={p.fits}', p.fits is False))

    # More slots than the whole expert table is a plan, not an error, and must not wrap
    hw = Hardware(name='1TB', vram_gb=1024.0, ram_gb=2048.0, ram_avail_gb=1800.0,
                  cards=[Card(0, '1TB', 1024.0, 1000.0)])
    p = plan(hw=hw, n_layers=n_layers, model=prof, speculation=True)
    total_experts = n_layers * int(prof.get('n_experts') or 384)
    report.add(Case('planner does not wrap past the expert table',
                    f'slots={p.expert_slots} table={total_experts}',
                    p.expert_slots >= 0))

    # MTP must be off below 16GB and on at 24GB: drafting costs more resident layers
    # than it repays on a small card, and the planner owns that decision.
    small = Hardware(name='8GB', vram_gb=8.0, ram_gb=47.0, ram_avail_gb=30.0,
                     cards=[Card(0, '8GB', 8.0, 7.6)])
    big = Hardware(name='24GB', vram_gb=24.0, ram_gb=64.0, ram_avail_gb=50.0,
                   cards=[Card(0, '24GB', 24.0, 22.8)])
    ps = plan(hw=small, n_layers=n_layers, model=prof, speculation=True)
    pb = plan(hw=big, n_layers=n_layers, model=prof, speculation=True)
    report.add(Case('planner keeps MTP off on an 8GB card',
                    f'speculation={ps.speculation}', ps.speculation is False))
    report.add(Case('planner enables MTP on a 24GB card',
                    f'speculation={pb.speculation}', pb.speculation is True))
