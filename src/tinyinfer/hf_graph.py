"""CUDA-graph decode loop for HuggingFace causal LMs.

At decode a 0.8B model needs ~10ms of GPU but ~30ms of Python: the same arithmetic costs
3-4x because ~1200 kernels per token are launched from Python one at a time. Capturing
the step once and replaying it removes that dispatcher cost entirely.

Three things make this work:

* native.launch passes torch's *current* stream instead of legacy stream 0. Stream 0 is
  not recorded by graph capture, so such a kernel runs for real during capture and is
  absent from every replay -- GatedDeltaNet's recurrent state then stops advancing and
  the second replay yields NaN.
* The cache must be snapshotted *recursively*. GatedDeltaNet keeps conv_states and
  recurrent_states in {slot: tensor} dicts; an attribute-level scan misses exactly the
  state that makes the cache advance, and every generated token shifts by one.
* Capture executes the step for real, so the snapshot is restored afterwards and the
  replay starts from the same state as the eager reference. Replay is verified against
  that reference before generating, on the same state.
"""
from __future__ import annotations

import time

import torch


def _cache_snapshot(cache):
    """Every mutable tensor a cache layer holds, including dict-valued state."""
    out = []

    def take(v):
        if torch.is_tensor(v):
            out.append((v, v.clone()))
        elif isinstance(v, dict):
            for item in v.values():
                take(item)
        elif isinstance(v, (list, tuple)):
            for item in v:
                take(item)

    for lay in getattr(cache, "layers", []):
        for _k, v in vars(lay).items():
            take(v)
    return out


def _cache_restore(snap):
    for orig, cp in snap:
        orig.copy_(cp)


class GraphDecoder:
    """Replay one decode step from a CUDA graph, verified against the eager path."""

    def __init__(self, model, ids_buf, pos_buf, logits_buf, cache, fwd):
        self.model = model
        self.ids_buf = ids_buf
        self.pos_buf = pos_buf
        self.logits_buf = logits_buf
        self.cache = cache
        self._fwd = fwd
        self.graph = None
        self.first_token = None

    @classmethod
    def from_prompt(cls, model, prompt_ids, keep=1, **extra):
        """Prefill, then build the graph for one decode step."""
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        cfg = model.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        V = int(cfg.vocab_size)
        T = prompt_ids.shape[1]
        with torch.inference_mode():
            out = model(input_ids=prompt_ids, use_cache=True, **extra)
            cache = out.past_key_values
            first = int(out.logits[0, -1].argmax())
            ids_buf = torch.zeros(1, 1, dtype=torch.long, device=prompt_ids.device)
            pos_buf = torch.zeros(1, dtype=torch.long, device=prompt_ids.device)
            logits_buf = torch.zeros(1, V, dtype=torch.float32, device=prompt_ids.device)
            ids_buf.copy_(out.logits[0, -1].argmax().view(1, 1))
            pos_buf.fill_(T)

            def fwd():
                lg = model(input_ids=ids_buf, past_key_values=cache, use_cache=True,
                           cache_position=pos_buf, logits_to_keep=keep).logits
                logits_buf.copy_(lg[:, -1].float())

            torch.cuda.synchronize()
            self = cls(model, ids_buf, pos_buf, logits_buf, cache, fwd)
            self.first_token = first
            return self

    def capture(self):
        """Capture the step and prove it reproduces the eager result."""
        with torch.inference_mode():
            snap = _cache_snapshot(self.cache)
            pos0 = int(self.pos_buf[0])
            tok0 = int(self.ids_buf[0, 0])

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._fwd()
            torch.cuda.synchronize()

            # Reference for exactly this state, through the ordinary path.
            _cache_restore(snap)
            self.pos_buf.fill_(pos0)
            self.ids_buf.fill_(tok0)
            self._fwd()
            torch.cuda.synchronize()
            ref = int(self.logits_buf.argmax())

            # Same state through the graph must agree, otherwise the whole loop is wrong.
            _cache_restore(snap)
            self.pos_buf.fill_(pos0)
            self.ids_buf.fill_(tok0)
            torch.cuda.synchronize()
            self.graph.replay()
            torch.cuda.synchronize()
            got = int(self.logits_buf.argmax())
            if got != ref:
                raise RuntimeError(
                    f"graph replay disagrees with eager on identical state ({got} vs {ref})")

            # Leave the state exactly as the prefill left it: the first replay consumes
            # tok0 at pos0 and produces the token that follows first_token.
            _cache_restore(snap)
            self.pos_buf.fill_(pos0)
            self.ids_buf.fill_(tok0)
            torch.cuda.synchronize()
        return self

    def next_token(self) -> int:
        with torch.inference_mode():
            self.graph.replay()
            tok = int(self.logits_buf.argmax())
            self.ids_buf.fill_(tok)
            self.pos_buf += 1
        return tok

    def generate(self, max_new=128, eos=None):
        gen = []
        t0 = time.perf_counter()
        if self.first_token is not None:
            gen.append(self.first_token)      # the prefill's own argmax
            self.first_token = None
        while len(gen) < max_new:
            tok = self.next_token()
            gen.append(tok)
            if eos is not None and tok == eos:
                break
        torch.cuda.synchronize()
        return gen, time.perf_counter() - t0
