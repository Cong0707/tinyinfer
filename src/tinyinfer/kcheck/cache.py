"""Cache eviction policies: the thing the 87% expert hit rate rests on.

Three policies share one interface (on_hit / on_insert / on_remove / evict_one). The
engine picks one by name and never looks inside, so a policy that returns a key it does
not hold, or evicts a pinned key, corrupts the expert cache silently -- the wrong bytes
come back and the layer computes something that is finite and wrong.
"""
from __future__ import annotations

from . import Case


def _policies():
    from ..loading import ARCPolicy, LRUPolicy, S3FIFOPolicy

    return [('lru', LRUPolicy), ('arc', ARCPolicy), ('s3fifo', S3FIFOPolicy)]


def register(report) -> None:
    try:
        policies = _policies()
    except Exception as exc:
        report.add(Case('cache_policy', f'unavailable: {exc!r}', False))
        return

    n_keys = 40
    for name, cls in policies:
        try:
            p = cls(capacity=n_keys)
        except Exception as exc:
            report.add(Case(f'{name} constructs', f'raised {exc!r}', False))
            continue

        # S3-FIFO promotes out of its 10% small queue only on a hit. Inserting 40 keys and
        # never touching them again evicts them all again -- correct for that policy and
        # not what the other two do. Hit each key once so the comparison is apples to
        # apples: that is also what the expert cache does (a blob is fetched and then read).
        keys = [(0, i) for i in range(n_keys)]
        for k in keys:
            p.on_insert(k)
            p.on_hit(k)
        held_n = len(p)
        report.add(Case(f'{name} retains keys after a hit', f'len={held_n}',
                        held_n >= max(1, n_keys // 2)))

        held = set(keys)
        evicted = []
        ok = True
        # drain until the policy reports empty; every victim must be one it actually holds
        for _ in range(n_keys * 2):
            got = p.evict_one(lambda k: False)
            if got is None:
                break
            if got not in held:
                ok = False
                break
            held.discard(got)
            evicted.append(got)
            p.on_remove(got)
        report.add(Case(f'{name} evicts only keys it holds',
                        f'{len(evicted)} evicted, {len(held)} left, len={len(p)}',
                        ok and len(p) == 0))
        report.add(Case(f'{name} returns None when empty',
                        '', p.evict_one(lambda k: False) is None))

        p2 = cls(capacity=8)
        for i in range(8):
            p2.on_insert((0, i))
        pinned = {(0, i) for i in range(7)}
        victim = p2.evict_one(lambda k: k in pinned)
        report.add(Case(f'{name} skips pinned keys',
                        f'victim={victim}', victim == (0, 7)))

        p3 = cls(capacity=8)
        for i in range(8):
            p3.on_insert((0, i))
        before = len(p3)
        p3.on_hit((0, 3))
        report.add(Case(f'{name} hit keeps the size',
                        f'{before}->{len(p3)}', len(p3) == before))
