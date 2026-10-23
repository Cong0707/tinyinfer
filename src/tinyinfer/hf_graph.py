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
    def from_prompt(cls, model, prompt_ids, keep=1, capacity=256, **extra):
        """Prefill, then build the graph for one decode step."""
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        cfg = model.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        V = int(cfg.vocab_size)
        T = prompt_ids.shape[1]
        with torch.inference_mode():
            # A STATIC cache, not DynamicCache. DynamicLayer.update concatenates
            # (torch.cat) which allocates every token; inside a capture that allocation is
            # frozen, and the measured result was a KV length stuck at 16 across four
            # replays while the logits kept changing -- the model only ever saw the
            # prefill and emitted one token forever. StaticLayer writes in place with
            # index_copy_ at a position taken from cache_position, so a captured step only
            # touches memory the graph can address.
            #
            # The cache also has to be told the config: without it the layer list is empty
            # and the first linear-attention layer raises IndexError. generate() builds
            # exactly this object internally.
            from transformers.cache_utils import StaticCache
            cap = int(prompt_ids.shape[1]) + int(capacity)
            cache = StaticCache(config=cfg, max_cache_len=cap)
            out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True, **extra)
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

            # One eager decode step before capturing. The prefilled state has just gone
            # through the chunked prefill path; a capture taken straight off it does not
            # reproduce the eager step (measured: graph 10 vs eager 220 at pos 18), while
            # the state after one ordinary decode step does. The token this step produces
            # is kept, so nothing is skipped.
            fwd()
            torch.cuda.synchronize()
            second = int(logits_buf.argmax())
            ids_buf.fill_(second)
            pos_buf += 1
            torch.cuda.synchronize()
            self = cls(model, ids_buf, pos_buf, logits_buf, cache, fwd)
            self.first_token = first
            self.second_token = second
            return self

    def capture(self, max_preroll: int = 3):
        """Capture the decode step, proving it reproduces the eager result.

        A capture taken at some cache positions does not reproduce the eager step; the
        failure depends on the position, not on the prompt (measured over five prompts:
        position immediately after prefill failed for two of them, and after at most two
        ordinary decode steps every one succeeded). So the routine advances the state one
        real decode step at a time and retries. Each step consumed this way still
        produces a real token, kept in \`pending\` and emitted before anything the graph
        generates.
        """
        self.pending = []
        with torch.inference_mode():
            for attempt in range(max_preroll + 1):
                snap = _cache_snapshot(self.cache)
                pos0 = int(self.pos_buf[0])
                tok0 = int(self.ids_buf[0, 0])

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._fwd()
                torch.cuda.synchronize()

                _cache_restore(snap)
                self.pos_buf.fill_(pos0)
                self.ids_buf.fill_(tok0)
                self._fwd()
                torch.cuda.synchronize()
                ref = int(self.logits_buf.argmax())

                _cache_restore(snap)
                self.pos_buf.fill_(pos0)
                self.ids_buf.fill_(tok0)
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                got = int(self.logits_buf.argmax())
                # Both paths above consumed pos0 and left the *same* logits. Put the
                # state back to before either ran; on success we re-apply exactly one step
                # below so the token stream has no gap.
                _cache_restore(snap)
                self.pos_buf.fill_(pos0)
                self.ids_buf.fill_(tok0)
                torch.cuda.synchronize()

                if got == ref:
                    # The graph reproduces the eager step here, so it is safe. Consume that
                    # step exactly once: its output is the first token this decoder emits,
                    # and the state ends one position ahead, which is what replay expects.
                    self._fwd()
                    torch.cuda.synchronize()
                    first_out = int(self.logits_buf.argmax())
                    self.graph = g
                    self.fallback_reason = None
                    self.capture_token = first_out
                    self.pos_buf.fill_(pos0 + 1)
                    self.ids_buf.fill_(first_out)
                    # Keep the state the step started from, plus the token it read. A
                    # later request rewinds to exactly this point and replays the step, so
                    # it reproduces the same first token instead of skipping it, and no
                    # second capture is needed (one costs ~137ms).
                    self.ready_state = _cache_snapshot(self.cache)
                    self.ready_pos = pos0 + 1
                    self.ready_token = first_out
                    torch.cuda.synchronize()
                    return self

                # Not aligned here: consume one real decode step and retry from there.
                _cache_restore(snap)
                self.pos_buf.fill_(pos0)
                self.ids_buf.fill_(tok0)
                self._fwd()
                torch.cuda.synchronize()
                self.pending.append(int(self.logits_buf.argmax()))
                self.ids_buf.fill_(self.pending[-1])
                self.pos_buf += 1
                torch.cuda.synchronize()
                del g
                torch.cuda.empty_cache()

            self.graph = None
            self.fallback_reason = f"capture disagreed after {max_preroll + 1} states"
        return self
    def next_token(self) -> int:
        with torch.inference_mode():
            if self.graph is None:            # eager fallback (see capture)
                self._fwd()
            else:
                self.graph.replay()
            tok = int(self.logits_buf.argmax())
            self.ids_buf.fill_(tok)
            self.pos_buf += 1
        return tok

    @property
    def using_graph(self) -> bool:
        return self.graph is not None

    def rewind(self) -> None:
        """Put the cache back to the state the capture left it in, ready for replay.

        A capture runs the step to prove the graph matches eager, so the cache ends one
        token ahead with that token sampled. Replaying from there is correct once; a second
        request must start from the same point, and re-running capture would cost ~137ms
        per completion.
        """
        state = getattr(self, "ready_state", None)
        if state is None:
            return
        with torch.inference_mode():
            # ready_state is the cache *before* the step the capture consumed, paired with
            # the token that step read. Restoring both and leaving pos at pre-step means the
            # next replay executes that same step, so its output is produced rather than
            # skipped -- which is what a fresh decoder emits via drain().
            _cache_restore(state)
            self.pos_buf.fill_(self.ready_pos)
            self.ids_buf.fill_(self.ready_token)

    def replay_step(self) -> None:
        """Advance one step and leave the logits in logits_buf, without choosing a token.

        next_token() argmaxes; a sampling backend has to see the distribution, so this
        splits the step in two.
        """
        with torch.inference_mode():
            if self.graph is None:
                self._fwd()
            else:
                self.graph.replay()

    def set_token(self, tok: int) -> None:
        """Feed the token the caller sampled back in, and advance the position."""
        self.ids_buf.fill_(int(tok))
        self.pos_buf += 1

    def drain(self) -> list[int]:
        """Tokens produced while preparing the graph, in emission order.

        Capturing consumes real decode steps (the pre-roll and the captured step itself),
        and each produced a genuine token. They come before anything a replay yields, so
        a streaming caller emits these first; the stream then has no gap and no repeat.
        """
        out: list[int] = []
        first = getattr(self, "first_token", None)
        if first is not None:
            out.append(first)
            self.first_token = None
        second = getattr(self, "second_token", None)
        if second is not None:
            out.append(second)
            self.second_token = None
        for tok in getattr(self, "pending", ()) or ():
            out.append(tok)
        self.pending = []
        cap = getattr(self, "capture_token", None)
        if cap is not None:
            out.append(cap)
            self.capture_token = None
        return out


    def generate(self, max_new=128, eos=None):
        gen = []
        t0 = time.perf_counter()
        if self.first_token is not None:
            gen.append(self.first_token)      # the prefill's own argmax
            self.first_token = None
        second = getattr(self, "second_token", None)
        if second is not None and len(gen) < max_new:
            gen.append(second)                # the pre-capture eager step's output
            self.second_token = None
        for tok in getattr(self, "pending", ()) or ():
            if len(gen) < max_new:            # produced by eager steps during capture
                gen.append(tok)
        # The capture itself consumed one step and produced `ref`; that token is real and
        # comes after everything the pending eager steps emitted.
        cap = getattr(self, "capture_token", None)
        if cap is not None and len(gen) < max_new:
            gen.append(cap)
            self.capture_token = None
        while len(gen) < max_new:
            tok = self.next_token()
            gen.append(tok)
            if eos is not None and tok == eos:
                break
        torch.cuda.synchronize()
        return gen, time.perf_counter() - t0