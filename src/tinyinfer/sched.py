"""Continuous batching: the serving loop that decodes several sequences at once.

The engine's own stream_generate() runs one sequence at a time. That throws away the
defining property of decode: a step reads every weight it touches exactly once, whether
it produces one token or eight. Eight sequences decoded together therefore cost one
weight read and produce eight tokens. On this checkpoint the read is ~12.7 GB per step,
43 ms of pure bandwidth at 295 GB/s; batching amortises that.

One class only. There were two (a slot-less prototype and this one) and keeping both
would mean two behaviours to maintain and two sets of bugs to find; the prototype is
gone and nothing used it.

What the loop does:

  admit    a request joins the running batch at a step boundary, without waiting for the
           current batch to drain -- static batching is what makes mixed load slow
  mix      prefill and decode share one forward call, because the weights for both are
           already resident; splitting them is two weight reads, i.e. the cost this
           module exists to remove
  retire   a finished sequence frees its KV slot for the next request immediately
  isolate  every sequence owns its position counter and stop set, so one finishing early
           never moves another's logits

The forward pass is the engine's own: batching is a property of the input tensor, not a
second implementation. It is NOT bit-identical to running the sequence alone -- cuBLAS
picks a different GEMM kernel for M=1 than for M>1 (measured: identical inputs, bf16
outputs one ulp apart), so any batched engine sees the same. The right acceptance test is
agreement of the sampled token, not equality of the logits.

"""

from __future__ import annotations

import time

import torch


class Engine:
    """The serving loop: admit, batch, forward once, sample, retire.

    Differs from BatchScheduler above in one respect that matters for a real server:
    it owns a KVPool, so a sequence that finishes frees its slot for the next request
    instead of the batch growing without bound. Admission is continuous -- a request
    joins the running batch at the next step boundary, not at the end of the current
    batch -- because waiting for a full batch to drain is what makes static batching
    slow under mixed load.
    """

    def __init__(self, engine, max_batch: int = 8):
        from .sequence import KVPool

        self.engine = engine
        self.max_batch = max(1, int(max_batch))
        self.pool = KVPool(self.max_batch)
        self.pending: list = []
        self.running: list = []
        self._fresh: set[int] = set()
        self.steps = 0
        self.tokens = 0
        self._step_ms = 0.0

    # ---- admission ----
    def submit(self, seq) -> None:
        self.pending.append(seq)

    def _admit(self) -> None:
        while self.pending and self.pool.free > 0:
            seq = self.pending.pop(0)
            if self.pool.acquire(seq) is None:
                self.pending.insert(0, seq)
                break
            self.running.append(seq)
            self._fresh.add(seq.id)

    def _retire(self) -> list:
        out = []
        keep = []
        for seq in self.running:
            finished = seq.done or seq.n_out >= seq.max_new_tokens
            if finished:
                self.pool.release(seq)
                out.append(seq)
            else:
                keep.append(seq)
        self.running = keep
        return out

    # ---- one step ----
    def _width(self) -> int:
        fresh = [s for s in self.running if s.id in self._fresh]
        return max((len(s.prompt) for s in fresh), default=1)

    def step(self) -> list:
        """Advance every running sequence one token; returns the finished ones."""
        from .nn import sample_logits

        self._admit()
        if not self.running:
            return []
        width = self._width()
        pad = self.engine.pad_id
        ids = torch.full((len(self.running), width), pad, dtype=torch.long,
                         device=self.engine.device)
        for i, s in enumerate(self.running):
            if s.id in self._fresh:
                ids[i, :len(s.prompt)] = torch.tensor(s.prompt, dtype=torch.long)
            else:
                ids[i, 0] = int(s.out[-1])
        # per-row positions: a fresh prompt's last real column is len(prompt)-1, a
        # decoding row's only column is its own next position
        pos = torch.tensor(
            [(len(s.prompt) - 1) if s.id in self._fresh else s.next_pos
             for s in self.running],
            dtype=torch.long, device=self.engine.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.engine.forward(ids, start_pos=int(pos.min().item()),
                                         positions=pos)
        self._step_ms = (time.perf_counter() - t0) * 1000.0
        self.steps += 1

        finished = []
        for i, s in enumerate(self.running):
            col = (len(s.prompt) - 1) if s.id in self._fresh else 0
            nid = sample_logits(logits[i, col], s.temperature, s.top_p, s.top_k)
            s.out.append(int(nid))
            self.tokens += 1
            self._fresh.discard(s.id)
            if s.stop_ids and nid in s.stop_ids:
                s.done = True
        return finished + self._retire()

    def run(self, budget_steps: int = 100000) -> list:
        """Serve until nothing is pending or running; returns every sequence."""
        served = list(self.pending) + list(self.running)
        for _ in range(budget_steps):
            self.step()
            if not self.running and not self.pending:
                break
        return served

    def stats(self) -> dict:
        d = {
            'steps': self.steps,
            'tokens': self.tokens,
            'running': len(self.running),
            'pending': len(self.pending),
            'last_step_ms': round(self._step_ms, 2),
        }
        d.update(self.pool.stats())
        return d