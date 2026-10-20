"""CPU-side context-aware expert prefetcher.

Routing is (mostly) a deterministic function of the current token and its short
context, so the predictor keeps four cheap tables:

  token     : token id            -> per-layer expert counters  (seen tokens)
  bigram    : (prev, token)       -> per-layer expert counters  (context)
  trigram   : (prev2, prev, tok)  -> per-layer expert counters
  temporal  : layer               -> previous token's expert set

predict(token) merges the table votes (weighted by table type and vote counts)
and falls back to temporal/hot sets. observe() updates everything online from
the real routing. Pure CPU, a few MB.

Expert choice is semantic; token tables capture that far better than branch
history hashing, and the merge keeps it robust for unseen tokens.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class RoutePredictorConfig:
    width: int = 8
    bigram_weight: float = 1.0
    token_weight: float = 0.8
    trigram_weight: float = 1.2
    temporal_weight: float = 0.6
    hot_weight: float = 0.3
    max_token_rows: int = 200_000
    max_gram_rows: int = 500_000


class RoutePredictor:
    def __init__(self, cfg: RoutePredictorConfig | None = None):
        self.cfg = cfg or RoutePredictorConfig()
        self.token: dict[int, dict[int, Counter]] = {}
        self.bigram: dict[tuple[int, int], dict[int, Counter]] = {}
        self.trigram: dict[tuple[int, int, int], dict[int, Counter]] = {}
        self.temporal: dict[int, list[int]] = {}
        self.hot: dict[int, Counter] = {}
        self.prevs: list[int] = []
        self.predicted = 0
        self.hit = 0
        self._pending = None

    def _bump(self, table: dict, key, routing: dict[int, list[int]], limit: int) -> None:
        row = table.get(key)
        if row is None:
            if len(table) >= limit:
                table.pop(next(iter(table)))
            row = {}
            table[key] = row
        for layer, experts in routing.items():
            c = row.get(layer)
            if c is None:
                c = row[layer] = Counter()
            c.update(experts)

    def observe(self, token: int, routing: dict[int, list[int]]) -> None:
        self._bump(self.token, int(token), routing, self.cfg.max_token_rows)
        prev1 = self.prevs[-1] if self.prevs else -1
        prev2 = self.prevs[-2] if len(self.prevs) >= 2 else -1
        self._bump(self.bigram, (prev1, int(token)), routing, self.cfg.max_gram_rows)
        self._bump(self.trigram, (prev2, prev1, int(token)), routing, self.cfg.max_gram_rows)
        for layer, experts in routing.items():
            self.temporal[layer] = list(experts)
            self.hot.setdefault(layer, Counter()).update(experts)
        if self._pending is not None:
            tok, pred = self._pending
            self._pending = None
            if tok == int(token):
                for layer, plist in pred.items():
                    actual = routing.get(layer)
                    if actual is None:
                        continue
                    self.predicted += 1
                    self.hit += int(bool(set(plist) & set(actual)))
        self.prevs.append(int(token))
        if len(self.prevs) > 8:
            self.prevs = self.prevs[-8:]

    def predict(self, token: int) -> dict[int, list[int]]:
        prev1 = self.prevs[-1] if self.prevs else -1
        prev2 = self.prevs[-2] if len(self.prevs) >= 2 else -1
        sources = []
        tri = self.trigram.get((prev2, prev1, int(token)))
        if tri:
            sources.append((self.cfg.trigram_weight, tri))
        bi = self.bigram.get((prev1, int(token)))
        if bi:
            sources.append((self.cfg.bigram_weight, bi))
        tok = self.token.get(int(token))
        if tok:
            sources.append((self.cfg.token_weight, tok))
        out: dict[int, list[int]] = {}
        layers = set()
        for _w, src in sources:
            layers.update(src.keys())
        layers.update(self.temporal.keys())
        for layer in layers:
            votes: Counter = Counter()
            total = 0.0
            for weight, src in sources:
                c = src.get(layer)
                if not c:
                    continue
                total += weight
                for e, n in c.items():
                    votes[e] += weight * n
            for e in self.temporal.get(layer, [])[: self.cfg.width]:
                votes[e] += self.cfg.temporal_weight * (total + 1.0)
            for e, n in self.hot.get(layer, Counter()).most_common(self.cfg.width):
                votes[e] += self.cfg.hot_weight * n
            if votes:
                picked = []
                for e, score in votes.most_common(self.cfg.width):
                    # only confident experts: at least 12% of the layer's total vote mass
                    if total > 0 and score >= 0.12 * sum(votes.values()):
                        picked.append(e)
                if picked:
                    out[layer] = picked
        self._pending = (int(token), {k: list(v) for k, v in out.items()})
        return out

    def stats(self) -> dict:
        return {
            "predicted": self.predicted,
            "hit": self.hit,
            "hit_rate": self.hit / self.predicted if self.predicted else 0.0,
            "token_rows": len(self.token),
            "bigram_rows": len(self.bigram),
            "trigram_rows": len(self.trigram),
        }
