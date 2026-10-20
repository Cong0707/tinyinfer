"""O5 speculation support: n-gram drafter (no learned weights).

Keeps a suffix index over the token history and proposes continuations for the
longest matching suffix. Used by a future verify loop; the official DSpark/MTP
heads are the model-native alternative.
"""
from __future__ import annotations

import collections


class NgramDrafter:
    """Suffix-matched n-gram proposer (1..max_n grams)."""

    def __init__(self, max_n: int = 5, max_entries: int = 4096):
        self.max_n = max_n
        self.max_entries = max_entries
        self._index: dict[tuple[int, ...], list[int]] = {}
        self._history: list[int] = []

    def observe(self, ids: list[int]) -> None:
        """Extend the history and index the new n-grams."""
        for tok in ids:
            self._history.append(tok)
            for n in range(1, self.max_n + 1):
                if len(self._history) >= n + 1:
                    key = tuple(self._history[-(n + 1):-1])
                    nxt = self._history[-1]
                    lst = self._index.setdefault(key, [])
                    if not lst or lst[-1] != nxt:
                        lst.append(nxt)
                        if len(lst) > 8:
                            del lst[0]
        if len(self._history) > self.max_entries:
            # keep the index but bound the stored history
            self._history = self._history[-self.max_entries:]

    def propose(self, history: list[int], k: int) -> list[int]:
        """Return up to k draft tokens after the longest matching suffix of history."""
        if k <= 0 or not history:
            return []
        for n in range(min(self.max_n, len(history)), 0, -1):
            key = tuple(history[-n:])
            seq = self._index.get(key)
            if seq:
                draft = list(seq[:k])
                # extend greedily through the chain while matches exist
                while len(draft) < k:
                    key2 = tuple((history + draft)[-n:])
                    nxt = self._index.get(key2)
                    if not nxt:
                        break
                    cand = nxt[0]
                    if draft and cand == draft[-1]:
                        break
                    draft.append(cand)
                return draft[:k]
        return []
