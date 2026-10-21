"""tinyinfer engine: full-model driver on top of adapters + tiers + ExpertStore.

Owns model lifecycle: embedding, block residency (lazy per-layer load), generation.
The fetch/dequant/eviction policy lives in loading.py; this is the orchestration layer.
"""
from __future__ import annotations

from pathlib import Path

from .adapters import ModelAdapter, open_adapter
from .config import TinyConfig
from .hc import hc_split_sinkhorn  # noqa: F401  (re-export for callers)
from .nn import rms_norm, sample_logits, softmax_sample
from .model import Block
from .tiers import OptLevel, TierConfig, plan

import torch
import torch.nn.functional as F


class Engine:
    @classmethod
    def from_hf(cls, source: str | Path, **kwargs) -> "Engine":
        """Load directly from a HF repo id / hub snapshot / flat checkpoint dir.

        This is the canonical entry point: no per-deployment model-dir plumbing.
        Local-only by default; pass local_files_only=False to plan.resolve_hf_snapshot
        (via open_adapter) once network downloads are acceptable.
        """
        return cls(open_adapter(source), **kwargs)

    def __init__(
        self,
        adapter: ModelAdapter | str,
        *,
        tier: OptLevel | TierConfig = OptLevel.BALANCED,
        expert_store=None,
        max_layers: int | None = None,
        allow_missing_layers: bool = False,
        device: str = "cpu",
        tokenizer=None,
        gpu_layers: int | None = None,
        layers: list[int] | None = None,
        head_on_cpu: bool = False,
    ):
        self.layer_ids = list(layers) if layers is not None else None
        self.device = device
        self._tokenizer = tokenizer
        self.hashes = None
        self.adapter = open_adapter(adapter) if isinstance(adapter, str) else adapter
        self.cfg = self.adapter.load_config()
        self.store = self.adapter.store
        if isinstance(tier, str):
            from .tiers import plan as _tier_plan
            tier = _tier_plan(OptLevel[tier.upper()])
        self.tier = plan(tier) if isinstance(tier, OptLevel) else tier
        if expert_store is None and self.tier.level >= OptLevel.BALANCED:
            # production tiers own their ExpertStore (RAM cache + VRAM slots)
            try:
                from .loading import ExpertStore
                # elastic RAM budget: size the expert cache from what is actually free,
                # so other agents/processes on the box do not cause MemoryError
                try:
                    import psutil
                    avail = psutil.virtual_memory().available
                    cache_budget = int(avail * 0.5)  # larger RAM cache = fewer disk misses in fetch
                except Exception:
                    cache_budget = int(self.tier.expert_cache_gb * (1 << 30))
                self.expert_store = ExpertStore.from_store(
                    self.store,
                    cache_bytes=min(16 << 30, max(512 << 20, cache_budget)),
                    policy="arc",
                    vram_slots=96 if device != "cpu" else 0,  # 96*18.8MB=1.8GB, covers a token's experts
                    device=device if device != "cpu" else "cpu",
                    direct=False,
                )
            except Exception as e:
                print(f"[tinyinfer] ExpertStore disabled: {e}")
                self.expert_store = expert_store
        else:
            self.expert_store = expert_store
        self.max_layers = self.cfg.n_layers if max_layers is None else min(max_layers, self.cfg.n_layers)
        self.allow_missing_layers = allow_missing_layers
        self.prefill_chunk = 2048
        # how many layers may stay resident on the device; the rest stream per forward
        if gpu_layers is not None:
            self.gpu_layers = gpu_layers
            self.head_gpu = gpu_layers <= 20  # keep the head off-card when many layers are resident
            if head_on_cpu:
                self.head_gpu = False
        elif device == "cpu":
            self.gpu_layers = 40
            self.head_gpu = False
        else:
            # dynamic VRAM budget: reserve embed + expert slots, then trade the 1.3GB output
            # head against resident layers (a streamed layer costs ~46ms/token, the head
            # costs ~65ms/token on CPU) -- keep the head on GPU unless few layers can stay.
            # measured on this box: ~0.44GB VRAM per resident fp8 layer (weights 0.19 +
            # allocator/state), embed 1.3GB, 48 expert slots 0.9GB. Stay under 7.0GB
            # so nothing pages out (paging made attn 4x slower).
            if self.tier.keep_weights_quantized:
                layer_gb, embed_gb, slot_gb, budget = 0.44, 1.3, 0.9, 7.0
            else:
                layer_gb, embed_gb, slot_gb, budget = 0.65, 1.3, 0.0, 6.0
            usable = budget - embed_gb - slot_gb
            self.head_gpu = (usable - 1.3) / layer_gb >= 20
            head_gb = 0.0 if self.head_gpu else 1.3
            room = usable - head_gb
            self.gpu_layers = min(40, max(1, int(room / layer_gb)))
        self._blocks: dict[int, Block] = {}
        self._last_routing: dict[int, list[int]] = {}
        self.enable_prefetch = True
        self.cache_wo_a = False
        self.profile = False
        # CPU-side learned expert prefetcher (context tables; ~95% hit rate measured)
        self.predictor = None
        if self.tier.level >= OptLevel.BALANCED:
            from .routes import RoutePredictor
            self.predictor = RoutePredictor()
        if tokenizer is not None and self.cfg.engram_layer_ids:
            # building the compressed token map walks the whole 129k vocab: only do it when
            # the engram tables are actually on disk
            shards = {self.store.shard_of(f"layers.{i}.engram.embed.weight") for i in self.cfg.engram_layer_ids}
            if all((self.store.model_dir / s).exists() for s in shards):
                from .engram import EngramHashes
                try:
                    self.hashes = EngramHashes(self.cfg, tokenizer)
                except Exception as e:  # weighted table missing / tokenizer mismatch: run without engram
                    print(f"[tinyinfer] engram disabled: {e}")
        self.embed = self.store.get("embed.weight").to(device)
        self.out_norm = self.store.get("norm.weight").to(device)
        # the 1.3GB output head: off-VRAM it costs ~40ms/token but frees 1.3GB
        self.head_on_cpu = device != "cpu" and not self.head_gpu
        self.head = self.store.get("head.weight") if self.head_on_cpu else self.store.get("head.weight").to(device)
        self.eps = self.cfg.rms_norm_eps

    # -- residency ---------------------------------------------------------
    def available_layers(self) -> int:
        """How many consecutive layers have all their shards fully on disk (size-checked).

        A safetensors header is readable long before the data segments finish
        downloading; loading such a shard segfaults inside CUDA. Compare the file
        size against header + max data offset instead.
        """
        import json
        import struct

        def shard_complete(shard: str) -> bool:
            path = self.store.model_dir / shard
            if not path.exists():
                return False
            try:
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > 200 * 1024 * 1024:
                        return False  # truncated/garbage header
                    raw = f.read(n)
                    if len(raw) != n:
                        return False
                    header = json.loads(raw)
                data_end = max(v["data_offsets"][1] for v in header.values() if "data_offsets" in v)
                return path.stat().st_size == 8 + n + data_end
            except Exception:
                return False

        n = 0
        for i in range(self.cfg.n_layers):
            names = [f"layers.{i}.attn.wq_a.weight", f"layers.{i}.ffn.gate.weight",
                     f"layers.{i}.ffn.experts.0.w1.weight"]
            try:
                shards = {self.store.shard_of(nm) for nm in names}
            except Exception:
                break
            if not all(shard_complete(s) for s in shards):
                break
            n = i + 1
        return n

    def block(self, layer_id: int) -> Block:
        b = self._blocks.get(layer_id)
        if b is None:
            b = Block(self.store, layer_id, self.cfg, expert_store=self.expert_store,
                      quant=self.tier.keep_weights_quantized, hashes=self.hashes,
                      cache_wo_a=self.cache_wo_a)
            if layer_id < self.gpu_layers and self.device != "cpu":
                b.to(self.device)
            self._blocks[layer_id] = b
        return b

    def _layer_device(self, b: Block) -> str:
        return getattr(b, "_device", "cpu")

    def reset(self) -> None:
        self._last_routing.clear()
        if self.hashes is not None:
            self.hashes.reset()
        for b in self._blocks.values():
            if hasattr(b.attn, "reset"):
                b.attn.reset()
            b.ffn._expert_cache.clear()

    # -- slot snapshots (multi-agent KV ownership) -------------------------
    def snapshot(self) -> dict:
        """Capture per-layer KV state so another slot can own the conversation."""
        snap = {}
        for i, b in self._blocks.items():
            a = b.attn
            entry = {"kv": None if a._kv_window is None else a._kv_window.clone()}
            ckv = getattr(a, "_compress_kv", None)
            entry["ckv"] = None if ckv is None else ckv.clone()
            comp = getattr(a, "compressor", None)
            if comp is not None:
                entry["pending"] = None if comp._pending_kv is None else (comp._pending_kv.clone(), comp._pending_score.clone())
            idx = getattr(a, "indexer", None)
            if idx is not None and getattr(idx, "_k_cache", None) is not None:
                entry["ik"] = idx._k_cache.clone()
            snap[i] = entry
        return snap

    def restore(self, snap: dict) -> None:
        for i, entry in snap.items():
            b = self.block(i)
            a = b.attn
            a._kv_window = None if entry.get("kv") is None else entry["kv"].clone()
            if hasattr(a, "_compress_kv"):
                a._compress_kv = None if entry.get("ckv") is None else entry["ckv"].clone()
            comp = getattr(a, "compressor", None)
            if comp is not None:
                pending = entry.get("pending")
                if pending is None:
                    comp._pending_kv = comp._pending_score = None
                else:
                    comp._pending_kv, comp._pending_score = pending[0].clone(), pending[1].clone()
            idx = getattr(a, "indexer", None)
            if idx is not None:
                idx._k_cache = None if entry.get("ik") is None else entry["ik"].clone()
            b.ffn._expert_cache.clear()

    # -- double-buffered streaming -----------------------------------------
    def _ensure_stream_buffers(self, layer_id: int):
        """Two flat device buffers per layer-kind, laid out from this layer's weights.

        The first streamed layer of each kind defines the layout (same kind = same
        weight names/shapes); later layers reuse the buffers via rotating copy-in.
        """
        b = self.block(layer_id)
        kind = (self.cfg.compress_ratios[layer_id] > 0,
                getattr(b.attn, "compressor", None) is not None,
                getattr(b.attn, "indexer", None) is not None)
        cache = getattr(self, "_stream_buffer_cache", None)
        if cache is None:
            self._stream_buffer_cache = cache = {}
        ent = cache.get(kind)
        if ent is None:
            # offsets only; shapes are resolved per layer at copy time
            layout = []
            off = 0
            for owner, name, t in b._weight_paths():
                n = t.numel() * t.element_size()
                off = (off + 255) // 256 * 256
                layout.append((owner, name, None, off, n))
                off += n
            bufs = [torch.empty(off, dtype=torch.uint8, device=self.device) for _ in range(2)]
            ent = cache[kind] = (layout, bufs)
        return ent

    def _stream_layers(self):
        """Layers that are re-copied every token (the tail beyond the resident set)."""
        if getattr(self, "_stream_set", None) is None:
            if self.layer_ids is not None:
                self._stream_set = set(self.layer_ids[self.gpu_layers:])
            else:
                self._stream_set = set(range(self.gpu_layers, self.max_layers))
        return self._stream_set

    def _stream_prefetch(self, layer_id: int, stream) -> None:
        """Copy a streamed layer's weights into its rotating buffer (copy stream).

        layout offsets come from the first layer of this kind; shapes always come
        from the CURRENT layer (the same kind can still be packed differently across
        the source/consumer boundary).
        """
        (layout, bufs) = self._ensure_stream_buffers(layer_id)
        b = self.block(layer_id)
        paths = b._weight_paths()
        if b._cpu_refs is None:
            b._cpu_refs = [t.pin_memory() if t.is_cpu and t.device.type == "cpu" else t
                           for (_, _, t) in paths]
        if len(paths) != len(layout):
            raise RuntimeError(f"L{layer_id}: {len(paths)} weights vs layout {len(layout)}")
        buf = bufs[layer_id % 2]
        with torch.cuda.stream(stream):
            for (owner, name, t), (_, _, _, off, _) in zip(paths, layout):
                buf[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape).copy_(t)
        for (owner, name, t), (_, _, _, off, _) in zip(paths, layout):
            setattr(owner, name, buf[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape))
        b._device = self.device

    # -- streaming slots ---------------------------------------------------
    def _ensure_slots(self):
        if getattr(self, "_stream_slots", None) is None:
            self._stream_slots = {}
        return self._stream_slots

    def _slot_for(self, layer_id: int):
        """Two rotating device slots per attention kind; a streamed layer copies its
        weights into one, so the H2D is reused instead of allocating per token."""
        slots = self._ensure_slots()
        b = self.block(layer_id)
        # a slot must match the exact weight layout: compress ratios plus whether the
        # layer owns a compressor / indexer (source layers carry them, consumers don't)
        kind = (self.cfg.compress_ratios[layer_id] > 0,
                getattr(b.attn, "compressor", None) is not None,
                getattr(b.attn, "indexer", None) is not None)
        pair = slots.get(kind)
        if pair is None:
            pair = slots[kind] = []
        while len(pair) < 2:
            blk = Block(self.store, layer_id, self.cfg, expert_store=self.expert_store,
                        quant=self.tier.keep_weights_quantized, hashes=None)
            blk.to(self.device)
            pair.append(blk)
        return pair[layer_id % 2]

    def _stream_into_slot(self, block_i: Block, slot: Block) -> None:
        if block_i._cpu_refs is None:
            block_i._cpu_refs = [t.pin_memory() if t.is_cpu and t.device.type == "cpu" else t
                                 for (_, _, t) in block_i._weight_paths()]
        dst = slot._weight_paths()
        for (owner, name, t), cpu in zip(dst, block_i._cpu_refs):
            t.copy_(cpu, non_blocking=True)

    def _prefetch_layer(self, layer_id: int, experts) -> None:
        """Warm RAM and stage the experts in VRAM slots (DMA overlaps the current layer)."""
        b = self.block(layer_id)
        b.ffn.prefetch(experts)
        store = getattr(b.ffn, "_expert_store", None)
        if store is not None and getattr(store, "prefetch_to_vram", None) is not None:
            try:
                store.prefetch_to_vram(layer_id, experts)
            except Exception:
                pass

    # -- forward -----------------------------------------------------------
    def _forward_chunk(self, ids: torch.Tensor, start_pos: int):
        self._last_hidden = None  # filled after the final layer (MTP stage-0 input)
        self._target_layers = set(self.cfg.dspark_target_layer_ids)
        self._target_hidden: dict[int, torch.Tensor] = {}
        if self.layer_ids is not None:
            layer_seq = self.layer_ids
            if self.allow_missing_layers:
                avail = self.available_layers()
                layer_seq = [i for i in layer_seq if i < avail]
        else:
            n = self.max_layers
            if self.allow_missing_layers:
                n = min(n, self.available_layers())
            layer_seq = list(range(n))
        h = self.embed[ids]
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32, device=h.device)
        pre_mix[:, :, 0] = 1.0
        hash_ids = None
        if self.hashes is not None:
            hash_ids = self.hashes(ids.reshape(-1).tolist(), start_pos)
        # expert prefetch: learned CPU predictor first, temporal routing as fallback
        pred = None
        tok_now = None
        if self.enable_prefetch and start_pos > 0:
            if self.predictor is not None and ids.numel() == 1:
                tok_now = int(ids.reshape(-1)[0].item())
                pred = self.predictor.predict(tok_now)
            if pred:
                for layer_i, experts in pred.items():
                    if layer_i in layer_seq and experts:
                        self._prefetch_layer(layer_i, experts)
            elif self._last_routing:
                for layer_i, ids_ in self._last_routing.items():
                    if layer_i in layer_seq:
                        self._prefetch_layer(layer_i, ids_)
        self._timings = []
        events = []
        for i in layer_seq:
            b = self.block(i)
            streamed = i in self._stream_layers()
            if streamed and self.device != "cpu":
                b.to(self.device)
            if self.profile:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            if b.engram is not None and hash_ids is not None:
                h = b.engram.forward(h, hash_ids[b.engram.hash_index])
            h, pre_mix = b.forward(h, start_pos, pre_mix)
            if i in self._target_layers:
                # MTP stage-0 main_proj expects the concatenated collapsed hiddens of
                # the dspark target layers ([37,38,39] -> [1, T, 3*dim]).
                self._target_hidden[i] = b.hc_pre(h, pre_mix).detach()
            if self.profile:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                events.append((i, ev0, ev1))
            if b.ffn._last_ids:
                self._last_routing[i] = b.ffn._last_ids
            if streamed and self.device != "cpu":
                b.to("cpu")  # free VRAM for the next streamed layer
        if self.predictor is not None and tok_now is not None and self._last_routing:
            self.predictor.observe(tok_now, dict(self._last_routing))
        self._last_hidden = h  # [1, T, hc, dim] before collapse (MTP main hidden)
        if self.profile:
            torch.cuda.synchronize()
            self._timings = [(i, a.elapsed_time(b)) for i, a, b in events]
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.eps)
        if self.head_on_cpu:
            h = h.cpu()
        return torch.nn.functional.linear(h, self.head).float()

    def forward(self, ids: torch.Tensor, start_pos: int = 0, chunk: int | None = None):
        """Chunked prefill: KV/hash state carries across chunks (attention uses start_pos)."""
        if ids.device.type != self.device and self.device != "cpu":
            ids = ids.to(self.device)
        b, T = ids.shape
        step = chunk or self.prefill_chunk
        if T <= step:
            return self._forward_chunk(ids, start_pos)
        logits = None
        for off in range(0, T, step):
            logits = self._forward_chunk(ids[:, off:off + step], start_pos + off)
        return logits

    def stream_generate(self, prompt_ids, max_new_tokens: int = 64, temperature: float = 0.0,
                        top_p: float = 1.0, top_k: int = 0, stop_ids=None, chunk: int | None = None):
        """Yield generated token ids one at a time (prefill once, then decode)."""
        from .nn import sample_logits

        ids = list(prompt_ids)
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0, chunk=chunk)
            for step in range(max_new_tokens):
                nid = sample_logits(logits[0, -1], temperature, top_p, top_k)
                if stop_ids and nid in stop_ids:
                    break
                yield nid
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=len(ids) + step)

    # -- speculative decoding (MTP/DSpark) ---------------------------------
    def load_mtp(self) -> bool:
        """Load the three DSpark stages if their shards are on disk."""
        from .mtp import load_mtp
        stages = load_mtp(self.store, self.cfg, quant=self.tier.keep_weights_quantized)
        if stages is None:
            return False
        for st in stages:
            st.embed = getattr(self, "embed", None)  # drafts reuse the main embedding
            st.head = getattr(self, "head", None)    # drafts reuse the main output head
            st.to(self.device)
        self.mtp = stages
        return True

    def speculative_generate(self, prompt_ids, max_new_tokens: int = 32, draft_len: int | None = None,
                             temperature: float = 0.0):
        """Draft with MTP, verify with the main model (greedy accept), rollback on mismatch.

        Greedy verification: a draft token is accepted when the main model's argmax at
        that position matches it; the first mismatch ends the run. KV is snapshotted
        before the verify forward and rolled back so the main model state stays exact.
        """
        from .nn import sample_logits
        if getattr(self, "mtp", None) is None and not self.load_mtp():
            raise RuntimeError("MTP weights are not available (model-00044..46 not downloaded?)")
        draft_len = draft_len or min(self.cfg.dspark_block_size, 4)
        ids = list(prompt_ids)
        out = []
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            while len(out) < max_new_tokens:
                nid = sample_logits(logits[0, -1], temperature)
                out.append(nid)
                if len(out) >= max_new_tokens:
                    break
                # main-model hidden for MTP stage 0 (KV already advanced by the token)
                _ = self.forward(torch.tensor([[nid]], device=self.device), len(ids))
                hidden = self._last_hidden
                pos = len(ids) + 1
                x, main_x, pre = None, None, None
                for st in self.mtp:
                    if st.stage == 0:
                        x, main_x = st.forward_embed(hidden, torch.tensor([nid], device=self.device))
                        pre = torch.zeros(1, x.size(1), self.cfg.hc_mult, device=self.device)
                        pre[:, :, 0] = 1.0
                    x, pre = st.forward(x, pos, pre, main_x if main_x is not None else x)
                # draft the next tokens with the main head on the MTP hidden state
                hd = x  # last stage hidden
                drafts = [nid]
                dlogits = self.head(hd.view(1, -1)) if self.head_on_cpu else torch.nn.functional.linear(
                    hd.view(1, -1).float(), self.head).float()
                for _ in range(draft_len - 1):
                    nxt = int(dlogits[0].argmax())
                    drafts.append(nxt)
                if len(drafts) < 2:
                    logits = self.forward(torch.tensor([[nid]], device=self.device), pos - 1)
                    ids.append(nid)
                    continue
                # verify the draft prefix with one main-model forward, then roll back
                snap = self.snapshot()
                cand = drafts[1:]
                vlogits = self.forward(torch.tensor([cand], device=self.device), pos)
                accept = 0
                for t in range(len(cand)):
                    if int(vlogits[0, t].argmax()) == cand[t]:
                        accept += 1
                    else:
                        break
                ids = ids + [nid] + cand[:accept]
                out.extend(cand[:accept])
                if accept == 0:
                    self.restore(snap)
                    logits = self.forward(torch.tensor([[nid]], device=self.device), pos - 1)
                else:
                    # keep the KV of the accepted prefix only
                    self.restore(snap)
                    logits = self.forward(torch.tensor([[nid]] + cand[:accept], device=self.device), pos - 1)
                if len(out) >= max_new_tokens:
                    break
        return out[:max_new_tokens]

    def generate(self, prompt_ids, max_new_tokens: int = 16, temperature: float = 0.0, eos_token_id=None,
                 top_p: float = 1.0, top_k: int = 0):
        """Greedy/sampled generation loop (prefill once, then decode step by step)."""
        ids = list(prompt_ids)
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            out = []
            for step in range(max_new_tokens):
                nid = int(sample_logits(logits[0, -1], temperature, top_p, top_k))
                out.append(nid)
                if eos_token_id is not None and nid == eos_token_id:
                    break
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=len(ids) + step)
        return out

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict:
        gpu = sum(1 for i in self._blocks if self._layer_device(self._blocks[i]) != "cpu")
        out = {
            "layers_built": len(self._blocks),
            "layers_on_gpu": gpu,
            "max_layers": self.max_layers,
            "n_layers": self.cfg.n_layers,
            "gpu_layers_budget": self.gpu_layers,
            "tier": self.tier.level.name,
            "device": self.device,
        }
        if self.expert_store is not None:
            out["experts"] = self.expert_store.stats()
        if torch.cuda.is_available() and self.device != "cpu":
            out["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        return out
