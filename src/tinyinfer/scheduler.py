"""Request scheduling as prepared-graph selection (port of Flare's model).

Flare treats a captured graph as a **nonpreemptible service quantum**: the scheduler
does not slice time arbitrarily, it chooses among quanta (prefill chunk, decode batch,
speculative width) and admits requests into their lanes. Fairness is accounted per
request over the service it actually received, with a continuation value so a live
decode stream is not starved by a burst of prefills.

This module is the part of that model that does not depend on CUDA graphs: a request
table, the quantum choice, and an EEVDF-style virtual-time account. It is used by the
server loop; `Engine.generate` keeps its synchronous contract for tests and the CLI.

Nothing here allocates GPU memory or touches the model. Keeping it separate is
deliberate: Flare's rule is that scheduling and cache policy are shared runtime
policy, never model-specific code.
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    prompt_tokens: int
    output_budget: int
    generated: int = 0
    consumed: int = 0           # tokens actually fed to the model
    arrival: float = field(default_factory=time.time)
    last_service: float = 0.0
    finished: bool = False
    # EEVDF accounting
    virtual_start: float = 0.0
    weight: float = 1.0

    @property
    def prompt_left(self) -> int:
        return max(0, self.prompt_tokens - self.consumed)

    @property
    def output_left(self) -> int:
        return max(0, self.output_budget - self.generated)

    @property
    def done(self) -> bool:
        return self.finished or (self.prompt_left == 0 and self.output_left == 0)


@dataclass(frozen=True)
class Quantum:
    """One prepared execution choice: a prefill wave or a decode batch."""
    kind: str           # "prefill" | "decode"
    capacity: int       # tokens for prefill, rows for decode
    duration: float     # measured/estimated seconds

    def score(self, work: float) -> float:
        """Useful work per second of a nonpreemptible quantum."""
        return work / max(1e-6, self.duration)


class Scheduler:
    """EEVDF-lite: pick the quantum that completes the most useful work per second,
    then admit the request with the smallest virtual start time."""

    def __init__(self, prefill_caps=(128, 256, 512, 1024, 2048, 4096),
                 decode_caps=(1, 2, 4), durations=None):
        self.prefill_caps = tuple(prefill_caps)
        self.decode_caps = tuple(decode_caps)
        self.durations = dict(durations or {})
        self._waiting: list[tuple[float, int, Request]] = []
        self._seq = itertools.count()
        self._requests: dict[int, Request] = {}
        self._vtime = 0.0
        self._service = 0.0

    # -- table -----------------------------------------------------------
    def submit(self, rid: int, prompt_tokens: int, output_budget: int, weight: float = 1.0) -> Request:
        r = Request(rid=rid, prompt_tokens=prompt_tokens, output_budget=output_budget,
                    weight=max(1e-3, weight), virtual_start=self._vtime)
        self._requests[rid] = r
        heapq.heappush(self._waiting, (r.virtual_start, next(self._seq), r))
        return r

    def get(self, rid: int) -> Request | None:
        return self._requests.get(rid)

    def finish(self, rid: int) -> None:
        r = self._requests.get(rid)
        if r is not None:
            r.finished = True

    def active(self) -> list[Request]:
        return [r for r in self._requests.values() if not r.done]

    # -- selection -------------------------------------------------------
    def _dur(self, q: Quantum) -> float:
        d = self.durations.get((q.kind, q.capacity))
        return float(d) if d else q.duration

    def next_quantum(self) -> tuple[Quantum | None, list[Request]]:
        """Choose the next quantum and the requests that occupy it."""
        pending = sorted(self.active(), key=lambda r: (r.virtual_start, r.arrival))
        if not pending:
            return None, []
        prefill = [r for r in pending if r.prompt_left > 0]
        decode = [r for r in pending if r.prompt_left == 0 and r.output_left > 0]

        # A decode stream that has been waiting longer than one quantum of service gets
        # priority: it is live output, and starving it shows up directly as latency.
        now = time.time()
        starved = [r for r in decode if (now - r.last_service) > self._quantum_hint()]
        if starved:
            cap = self._pick_cap(len(starved), self.decode_caps)
            return Quantum("decode", cap, self._dur(Quantum("decode", cap, 0.01))), starved[:cap]
        if decode:
            cap = self._pick_cap(len(decode), self.decode_caps)
            return Quantum("decode", cap, self._dur(Quantum("decode", cap, 0.01))), decode[:cap]
        if prefill:
            want = sum(r.prompt_left for r in prefill[:4])
            cap = self._pick_cap(want, self.prefill_caps)
            chosen, total = [], 0
            for r in prefill:
                if total >= cap:
                    break
                chosen.append(r)
                total += r.prompt_left
            return Quantum("prefill", cap, self._dur(Quantum("prefill", cap, 0.3))), chosen
        return None, []

    def _pick_cap(self, want: int, caps) -> int:
        for c in caps:
            if c >= want:
                return c
        return caps[-1] if caps else max(1, want)

    def _quantum_hint(self) -> float:
        return max(self.durations.values(), default=0.05) * 2.0

    # -- accounting ------------------------------------------------------
    def charge(self, quantum: Quantum, served: list[Request], tokens: int) -> None:
        """Account service: EEVDF virtual time grows by duration / weight."""
        if not served:
            return
        d = self._dur(quantum)
        share = d / len(served)
        for r in served:
            r.virtual_start += share / r.weight
            r.last_service = time.time()
            if quantum.kind == "prefill":
                took = min(r.prompt_left, max(1, tokens // max(1, len(served))))
                r.consumed += took
            else:
                r.generated += 1
                r.consumed += 1
            heapq.heappush(self._waiting, (r.virtual_start, next(self._seq), r))
        self._vtime = min((r.virtual_start for r in self._requests.values() if not r.done),
                          default=self._vtime)
        self._service += d

    def stats(self) -> dict:
        return {"requests": len(self._requests), "active": len(self.active()),
                "vtime": round(self._vtime, 4), "service_s": round(self._service, 3)}
