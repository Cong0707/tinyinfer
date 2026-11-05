"""VRAM expert slot lifecycle: the state machine 24GB+ cards depend on.

On an 8GB card the planner gives 0 slots, so this path never runs here -- but it is the
whole reason a 24GB card can serve a 296GB expert table without touching PCIe. It is
exercised directly here instead of through an engine that has no room for it.
"""
from __future__ import annotations

import torch

from . import Case


def _slot_lifecycle(report):
    try:
        from ..loading import VramSlotCache
    except Exception as exc:
        report.add(Case('vram_slot', f'unavailable: {exc!r}', False))
        return
    if not torch.cuda.is_available():
        report.add(Case('vram_slot', 'no CUDA device', False))
        return

    blob = 1 << 20
    c = VramSlotCache(4, blob, device='cuda:0')
    tag = 0xAB

    # first acquire of a key is not ready: the caller has to fill the slot
    slot, ready = c.acquire(0, 42)
    got_ready = ready
    c.copy_in(slot, torch.full((blob,), tag, dtype=torch.uint8))
    c.publish(slot)
    intact = bool((c.buffer(slot) == tag).all().item())
    c.release(0, 42)
    report.add(Case('vram_slot first acquire is not ready', '', got_ready is False))
    report.add(Case('vram_slot copy_in survives publish', '', intact))

    # second acquire is a hit and must return the SAME bytes
    slot2, ready2 = c.acquire(0, 42)
    same_slot = slot2 == slot
    still_tag = bool((c.buffer(slot2) == tag).all().item())
    c.release(0, 42)
    report.add(Case('vram_slot hit returns same slot', '', same_slot))
    report.add(Case('vram_slot hit is ready and holds the data', '',
                    ready2 is True and still_tag))

    # eviction: with 2 slots, the third key must displace the oldest and keep the other
    c2 = VramSlotCache(2, blob, device='cuda:0')
    for e in (1, 2):
        s, _ = c2.acquire(0, e)
        c2.copy_in(s, torch.full((blob,), e, dtype=torch.uint8))
        c2.publish(s)
        c2.release(0, e)
    s3, ready3 = c2.acquire(0, 3)
    report.add(Case('vram_slot evicts to make room', '',
                    s3 >= 0 and ready3 is False and c2.evictions == 1))
    report.add(Case('vram_slot keeps the other key resident', '',
                    c2.has(0, 2) and c2.has(0, 3) and not c2.has(0, 1)))

    # refcount: a key held by two acquires must not be evicted under them
    c3 = VramSlotCache(1, blob, device='cuda:0')
    s, _ = c3.acquire(0, 7)
    c3.copy_in(s, torch.full((blob,), 7, dtype=torch.uint8))
    c3.publish(s)
    s_a, r_a = c3.acquire(0, 7)
    s_b, r_b = c3.acquire(0, 7)
    # a different key cannot evict it: both refs are live
    s_c, r_c = c3.acquire(0, 8)
    report.add(Case('vram_slot refuses to evict a held slot', '',
                    s_c == -1 and r_c is False))
    for _ in range(3):
        c3.release(0, 7)


def _slot_placement(report):
    try:
        from ..loading import VramSlotCache
    except Exception as exc:
        report.add(Case('vram_slot_devices', f'unavailable: {exc!r}', False))
        return
    if not torch.cuda.is_available():
        return
    c = VramSlotCache(4, 1 << 20, device='cuda:0')
    c.set_devices(['cuda:0'], [1])
    devs = [c.slot_device(i) for i in range(4)]
    report.add(Case('vram_slot_devices single card', '', set(devs) == {0}))


def register(report) -> None:
    _slot_lifecycle(report)
    _slot_placement(report)
