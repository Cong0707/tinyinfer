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
        # hardware-derived plan (VRAM/RAM budget -> resident layers, slots, cache, chunk)
        from .resources import plan as _hw_plan
        self.hw_plan = _hw_plan(None, self.cfg.n_layers) if device != "cpu" else None
        if self.hw_plan is not None:
            print(f"[tinyinfer] {self.hw_plan.reason}", flush=True)
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
                hw = self.hw_plan
                self.expert_store = ExpertStore.from_store(
                    self.store,
                    reader_threads=12,
                    cache_bytes=min(16 << 30, max(512 << 20, cache_budget)) if hw is None else hw.cache_bytes,
                    policy="arc",
                    vram_slots=(96 if device != "cpu" else 0) if hw is None else hw.expert_slots,
                    device=device if device != "cpu" else "cpu",
                    # direct=NO_BUFFERING reads: measured 6.3GB/s at 6 threads vs
                    # 0.26GB/s through the Windows file cache (which serialises the
                    # multi-threaded pread path on a big-blob stream)
                    direct=True,
                )
            except Exception as e:
                print(f"[tinyinfer] ExpertStore disabled: {e}")
                self.expert_store = expert_store
        else:
            self.expert_store = expert_store
        self.max_layers = self.cfg.n_layers if max_layers is None else min(max_layers, self.cfg.n_layers)
        self.allow_missing_layers = allow_missing_layers
        self.prefill_chunk = 2048  # overridden by the hardware plan on cuda
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
            # hardware-derived plan (resources.py): adaptive VRAM/RAM budgeting for any
            # card size (8GB laptop .. 80GB desktop .. unified-memory boxes), no manual flags.
            hw = self.hw_plan
            import os as _os
            # overrides for experiments: residency 0 makes every layer stream through
            # the flat pair, which is more H2D but frees the VRAM the pair needs
            self.gpu_layers = max(0, int(_os.environ.get("TI_GPU_LAYERS", hw.gpu_layers)))
            self.head_gpu = (not hw.head_on_cpu) and self.gpu_layers > 20
            self.prefill_chunk = int(_os.environ.get("TI_CHUNK", hw.prefill_chunk))
            # placement: while most layers stream, small activations round-trip every
            # op; keep HC-style tiny ops on the host (big-VRAM boxes stay all-GPU).
            from .placement import small_op_on_cpu
            self.small_ops_on_cpu = small_op_on_cpu(self.gpu_layers, self.max_layers)
        self._blocks: dict[int, Block] = {}
        self._last_routing: dict[int, list[int]] = {}
        # expert prefetch is opt-in: with the RAM cache holding ~600 of 15360 experts
        # the prediction misses often enough that the prefetch and the fetch read the
        # same blob twice (measured 5.3GB/step against 3.5GB of unique data)
        self.enable_prefetch = bool(int(__import__("os").environ.get("TI_PREFETCH", "0")))
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

    def _rt_for(self, layer_id: int):
        """Handoff slots for a layer: the nearest kv_source at or below it.

        Chunk-major execution reuses one process singleton because layers run in order;
        layer-major needs a separate instance per source so followers of different
        chunks cannot observe a stale neighbour's state.
        """
        from .compressor import SharedAttentionRuntime, shared_attn

        if not getattr(self, "_layer_major_mode", False):
            return shared_attn
        cache = getattr(self, "_rt_cache", None)
        if cache is None:
            self._rt_cache = cache = {}
        if layer_id in cache:
            return cache[layer_id]
        sources = [s for s in self.cfg.kv_source_layer_ids if s <= layer_id]
        src = max(sources) if sources else -1
        rt = cache.get(src)
        if rt is None:
            rt = cache[src] = SharedAttentionRuntime()
        cache[layer_id] = rt
        return rt

    def block(self, layer_id: int) -> Block:
        b = self._blocks.get(layer_id)
        if b is None:
            b = Block(self.store, layer_id, self.cfg, expert_store=self.expert_store,
                      quant=self.tier.keep_weights_quantized, hashes=self.hashes,
                      cache_wo_a=self.cache_wo_a, expert_topk=self.tier.expert_topk,
                      rt=self._rt_for(layer_id))
            b.hc_post_on_cpu = getattr(self, "small_ops_on_cpu", False)
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
        """Two flat device buffers per EXACT weight signature (v5).

        v4 grouped by (ratio>0, compressor, indexer) and broke at the L20 boundary
        (ratio=1 kv_source has a different weight count than the ratio=1 consumers).
        v5 keys the cache by the full (name, shape) signature, so layers that can
        share a flat buffer are exactly the ones whose weight list is identical.
        """
        b = self.block(layer_id)
        paths = b._weight_paths()
        sig = tuple((name, tuple(t.shape)) for (_, name, t) in paths)
        cache = getattr(self, "_stream_buffer_cache", None)
        if cache is None:
            self._stream_buffer_cache = cache = {}
        ent = cache.get(sig)
        if ent is None and len(cache) >= 1:
            # VRAM guard: one signature group x 2 buffers (2 x 190MB = 380MB); later
            # groups fall back to the synchronous per-layer copy.
            return None, None
        if ent is None:
            layout = []
            off = 0
            for owner, name, t in paths:
                n = t.numel() * t.element_size()
                off = (off + 255) // 256 * 256
                layout.append((owner, name, None, off, n))
                off += n
            bufs = [torch.empty(off, dtype=torch.uint8, device=self.device) for _ in range(2)]
            ent = cache[sig] = (layout, bufs)
        return ent

    def _stream_copy_sync(self, layer_id: int) -> None:
        """Copy a streamed layer's weights into its rotating buffer on the compute stream."""
        (layout, bufs) = self._ensure_stream_buffers(layer_id)
        b = self.block(layer_id)
        paths = b._weight_paths()
        buf = bufs[layer_id % 2]
        for (owner, name, t), (_, _, _, off, _) in zip(paths, layout):
            buf[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape).copy_(t)
            setattr(owner, name, buf[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape))
        if getattr(b, "engram", None) is not None:
            b.engram.to(self.device)
        b._device = self.device

    # -- flat streaming (v6) -----------------------------------------------
    # Every streamed layer reuses ONE pair of device buffers sized to the largest
    # layer, and every layer's host weights are packed once into a pinned flat
    # buffer. A decode step then issues one 442MB pinned H2D per layer (16ms at
    # pinned bandwidth) instead of ~31 pageable per-tensor copies that each cost a
    # host round-trip; the next layer is copied on a side stream while the current
    # one computes, so the transfer hides behind the GPU.
    def _flat_enabled(self) -> bool:
        st = getattr(self, "_flat_state", None)
        if st == "off":
            return False
        if st is None:
            try:
                self._flat_setup()
            except Exception as exc:  # pinned-pool exhaustion, OOM: stay on the old path
                print(f"[tinyinfer] flat streaming off: {exc!r}", flush=True)
                self._flat_state = "off"
                return False
        return True

    def _warm_weights(self, layers) -> None:
        """Read every streamed layer's weights through an 8-way pool first.

        ShardStore.get is one open+seek+read per tensor (31 per layer), which measured
        ~1.2GB/s on this box; the same bytes through 8 threads saturate far better and
        leave the block constructors reading from the OS cache.
        """
        import itertools
        from concurrent.futures import ThreadPoolExecutor

        names: list[str] = []
        try:
            wm = self.store.weight_map
        except AttributeError:
            return
        for i in layers:
            pre = f"layers.{i}."
            exp = pre + "ffn.experts."
            # non-expert weights only: the routed experts are 12.75GB/layer and are
            # streamed per token, warming them here would just OOM the box
            names.extend(n for n in wm if n.startswith(pre) and not n.startswith(exp))
        if not names:
            return
        import time as _time
        t0 = _time.time()
        total = 0
        pool = ThreadPoolExecutor(max_workers=8)
        it = iter(names)
        try:
            while True:
                batch = list(itertools.islice(it, 8))
                if not batch:
                    break
                for t in pool.map(self.store.get, batch):
                    total += t.numel() * t.element_size()
        finally:
            pool.shutdown(wait=True)
        dt = _time.time() - t0
        print(f"[tinyinfer] prewarm {total/1e9:.1f}GB of layer weights in {dt:.1f}s "
              f"= {total/1e9/max(dt,1e-6):.1f}GB/s", flush=True)

    def _flat_setup(self) -> None:
        import os

        if os.environ.get("TI_FLAT", "0") != "1":
            raise RuntimeError("TI_FLAT is opt-in (set TI_FLAT=1)")
        streamed = sorted(self._stream_layers())
        if not streamed:
            raise RuntimeError("no streamed layers")
        lay: dict[int, tuple[list[tuple[int, int]], int]] = {}
        sizes: dict[int, int] = {}
        for i in streamed:
            paths = self.block(i)._weight_paths()
            offs: list[tuple[int, int]] = []
            off = 0
            for (_, _, t) in paths:
                n = t.numel() * t.element_size()
                off = (off + 255) // 256 * 256
                offs.append((off, n))
                off += n
            lay[i] = (offs, off)
            sizes[i] = off
        # No second host copy: Block.to() already keeps a pinned ref for every weight
        # (_cpu_refs, 15.5GB on this checkpoint). A packed flat copy on top of that is
        # another 15.5GB, which pushed the box into paging (rss 21.6GB vs 14.6GB
        # available). Instead the per-tensor pinned copies go straight into the shared
        # pair: still async, still one H2D per layer, no extra host memory.
        keep: list[int] = []
        mx = max(sizes.values())
        # Three rotating buffers. With two, the H2D for layer k+1 starts only when
        # layer k begins, and a 719MB transfer (27ms) is slower than one layer of
        # compute (~15ms), so it can never hide. Three let layer k compute while
        # k+1 and k+2 stream in.
        bufs = [torch.empty(mx, dtype=torch.uint8, device=self.device) for _ in range(3)]
        pinned_gb = 0.0
        for i in streamed:
            b = self.block(i)
            paths = b._weight_paths()
            refs = []
            for (_, _, t) in paths:
                if t.is_cpu and not t.is_pinned():
                    try:
                        t = t.pin_memory()
                    except Exception:
                        pass
                refs.append(t)
            b._cpu_refs = refs
            pinned_gb += sum(t.numel() * t.element_size() for t in refs if t.is_pinned()) / 2 ** 30
            keep.append(i)
        self._flat_state = (lay, bufs, None)
        self._flat_ev = {}
        self._copy_stream = torch.cuda.Stream(device=self.device)
        print(f"[tinyinfer] flat streaming: {len(keep)} layers, {mx/2**20:.0f}MBx3 VRAM, "
              f"{pinned_gb:.1f}GB pinned source (no extra host copy)", flush=True)

    def _flat_apply(self, layer: int, buf_idx: int) -> None:
        lay, bufs, host = self._flat_state
        b = self.block(layer)
        offs, total = lay[layer]
        buf = bufs[buf_idx]
        if host is None:  # per-tensor copies from the pinned refs Block already holds
            for (owner, name, t), (o, n), src in zip(b._weight_paths(), offs, b._cpu_refs):
                buf[o:o + n].view(t.dtype).view(t.shape).copy_(src, non_blocking=True)
        else:
            buf[:total].copy_(host[layer], non_blocking=True)
        for (owner, name, t), (o, n) in zip(b._weight_paths(), offs):
            setattr(owner, name, buf[o:o + n].view(t.dtype).view(t.shape))
        if getattr(b, "engram", None) is not None:
            b.engram.to(self.device)
        b._device = self.device

    def _flat_prefetch(self, layer: int, buf_idx: int) -> None:
        if layer not in self._flat_state[0]:
            return
        ev = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            self._flat_apply(layer, buf_idx)
            ev.record(self._copy_stream)
        self._flat_ev[layer] = ev

    def _flat_wait(self, layer: int, buf_idx: int) -> None:
        import time as _t
        t0 = _t.perf_counter()
        try:
            if layer not in self._flat_state[0]:
                self.block(layer).to(self.device)
                return
            ev = self._flat_ev.pop(layer, None)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)
            else:
                self._flat_apply(layer, buf_idx)
        finally:
            self._h2d_wait_ms = getattr(self, "_h2d_wait_ms", 0.0) + (_t.perf_counter() - t0) * 1e3

    def _stream_layers(self):
        """Layers that are re-copied every token (the tail beyond the resident set)."""
        if getattr(self, "_stream_set", None) is None:
            if self.layer_ids is not None:
                self._stream_set = set(self.layer_ids[self.gpu_layers:])
            else:
                self._stream_set = set(range(self.gpu_layers, self.max_layers))
        return self._stream_set

    def _stream_prefetch(self, layer_id: int, stream):
        """Async-copy a streamed layer's weights into its rotating buffer (copy stream).

        v5: the buffer is keyed by the exact weight signature, so layout and the
        current layer always agree (L20 has its own buffers). Pinned CPU refs make
        the H2D run at pinned bandwidth. Returns the CUDA event to wait on before
        the layer computes.
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
        ev = torch.cuda.Event()
        with torch.cuda.stream(stream):
            for (owner, name, t), cpu, (_, _, _, off, n) in zip(paths, b._cpu_refs, layout):
                buf[off:off + n].view(t.dtype).view(t.shape).copy_(cpu, non_blocking=True)
            ev.record(stream)
        for (owner, name, t), (_, _, _, off, n) in zip(paths, layout):
            setattr(owner, name, buf[off:off + n].view(t.dtype).view(t.shape))
        if getattr(b, "engram", None) is not None:
            b.engram.to(self.device)
        return ev

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
                        quant=self.tier.keep_weights_quantized, hashes=None,
                        expert_topk=self.tier.expert_topk, rt=self._rt_for(layer_id))
            blk.hc_post_on_cpu = getattr(self, "small_ops_on_cpu", False)
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
        """Stage the experts for a later layer: one disk read, straight into a VRAM slot.

        The old version called both the RAM prefetch and the VRAM prefetch, so every
        blob was read from disk twice and the slot pool (measured 0 hits) was refilled
        with data that was never consumed.
        """
        b = self.block(layer_id)
        # RAM only. Staging into VRAM here filled the 24-slot pool with 186 predicted
        # blobs per token and evicted the very ones the compute thread was about to
        # use (measured: 2.4k loads, 1 hit). Slots are now H2D staging for the layer
        # being executed, and this thread just warms the RAM cache ahead of it.
        b.ffn.prefetch(experts)

    # -- forward -----------------------------------------------------------
    def _forward_chunk(self, ids: torch.Tensor, start_pos: int):
        self._last_hidden = None  # filled after the final layer (MTP stage-0 input)
        self._target_layers = set(self.cfg.dspark_target_layer_ids)
        self._target_hidden: dict[int, torch.Tensor] = {}
        # v5 streamed-layer double buffering state (lazily initialised; at most two
        # signature groups get flat buffers, the rest fall back to the sync copy).
        if not hasattr(self, "_v5_enabled"):
            # 8GB card: even two signature groups of flat buffers push VRAM past the
            # physical limit (measured 9330MB peak -> WDDM paging), so keep it off.
            self._v5_enabled = bool(int(__import__("os").environ.get("TI_V5", "0")))
            self._v5_stream = None
            self._v5_pending: dict[int, object] = {}
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
            source = pred or self._last_routing
            if source:
                # only the first layers: the VRAM slot pool holds ~1-2 layers of top-6,
                # so prefetching all 40 layers evicts every slot before it is consumed
                # (measured: 0 slot hits, every read repeated on the compute thread).
                # The rest is prefetched rolling inside the layer loop below.
                for layer_i in layer_seq[:2]:
                    if source.get(layer_i):
                        self._prefetch_layer(layer_i, source[layer_i])
        self._timings = []
        events = []
        if self._v5_enabled and self._v5_stream is None:
            self._v5_stream = torch.cuda.Stream(device=self.device)
        use_flat = self.device != "cpu" and not self._v5_enabled and self._flat_enabled()
        streamed_seq = [i for i in layer_seq if i in self._stream_layers()] if self._v5_enabled else []
        flat_seq = [i for i in layer_seq if i in self._stream_layers()] if use_flat else []
        flat_pos = {j: k for k, j in enumerate(flat_seq)}
        for pos_i, i in enumerate(layer_seq):
            b = self.block(i)
            streamed = i in self._stream_layers()
            if streamed and self.device != "cpu":
                if use_flat:
                    k = flat_pos[i]
                    self._flat_wait(i, k % 3)
                    for d in (1, 2):
                        if k + d < len(flat_seq):
                            nxt = flat_seq[k + d]
                            if nxt not in self._flat_ev:
                                self._flat_prefetch(nxt, (k + d) % 3)
                    if k + 2 < len(flat_seq) and self.enable_prefetch:
                        # RAM prefetch two layers ahead: the read has ~2 layers of compute
                        # time to land, and it never touches the VRAM slot pool
                        ahead = flat_seq[k + 2]
                        r = self._last_routing.get(ahead)
                        if r:
                            self._prefetch_layer(ahead, r)
                elif self._v5_enabled and streamed_seq:
                    try:
                        ev = self._v5_pending.pop(i, None)
                        if ev is not None:
                            torch.cuda.current_stream().wait_event(ev)
                        else:
                            if self._ensure_stream_buffers(i)[0] is None:
                                raise RuntimeError("buffer budget exceeded")  # sync fallback
                            self._stream_copy_sync(i)
                        # prefetch the next streamed layer into the other parity buffer
                        if i in streamed_seq:
                            k = streamed_seq.index(i)
                            if k + 1 < len(streamed_seq):
                                nxt = streamed_seq[k + 1]
                                if (nxt % 2) != (i % 2) and nxt not in self._v5_pending:
                                    self._v5_pending[nxt] = self._stream_prefetch(nxt, self._v5_stream)
                    except Exception:
                        self._v5_enabled = bool(int(__import__("os").environ.get("TI_V5", "0")))
                        b.to(self.device)
                else:
                    b.to(self.device)
            if self.profile:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            if b.engram is not None and hash_ids is not None:
                h = b.engram.forward(h, hash_ids[b.engram.hash_index])
            h, pre_mix = b.forward(h, start_pos, pre_mix)
            if i in self._target_layers:
                # MTP stage-0 main_proj expects the concat of the dspark target layers'
                # attn_norm outputs (official dflash_input), [1, T, 3*dim].
                cap = getattr(b, "_last_attn_norm", None)
                self._target_hidden[i] = (cap if cap is not None else b.hc_pre(h, pre_mix)).detach()
            if self.profile:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                events.append((i, ev0, ev1))
            if b.ffn._last_ids:
                self._last_routing[i] = b.ffn._last_ids
            if streamed and self.device != "cpu" and not use_flat:
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

    def prefill_layer_major(self, ids: torch.Tensor, max_chunk: int | None = None) -> torch.Tensor:
        """Layer-major prefill: stream each layer once for the whole prompt.

        The chunk-major path (forward) re-streams all 40 layers per chunk, so a long
        prompt pays the layer H2D and the expert reads once per chunk (measured
        prefill 64 = 314s at chunk 8, i.e. hours for 128K). Here the loop order is
        swapped: activations for every chunk sit in host RAM, each layer is moved to
        the device once, applied to all chunks, then freed. Only the activation list
        grows with the prompt (T x hc x dim x 2B = 5.2GB at 128K, fits in 47GB).
        """
        if ids.device.type != self.device and self.device != "cpu":
            ids = ids.to(self.device)
        b, T = ids.shape
        step = max_chunk or self.prefill_chunk
        # per-kv-source handoff slots for the whole pass (blocks are constructed lazily,
        # so the flag has to be set before any block() call below).
        self._layer_major_mode = True
        self._rt_cache = {}
        try:
            return self._prefill_layer_major_impl(ids, T, step)
        finally:
            self._layer_major_mode = False
            self._rt_cache = {}

    def _prefill_layer_major_impl(self, ids, T: int, step: int):
        if not hasattr(self, "_v5_enabled"):
            self._v5_enabled = bool(int(__import__("os").environ.get("TI_V5", "0")))
        # layer sequence identical to _forward_chunk
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
        # 1) embed every chunk once; keep the activation on the host between layers
        acts = []
        for off in range(0, T, step):
            chunk_ids = ids[:, off:off + step]
            h = self.embed[chunk_ids]
            h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
            pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32, device=h.device)
            pre_mix[:, :, 0] = 1.0
            hash_ids = None
            if self.hashes is not None:
                hash_ids = self.hashes(chunk_ids.reshape(-1).tolist(), off)
            acts.append([h.cpu(), pre_mix.cpu(), hash_ids, off])
        # 2) per layer: one H2D, apply to every chunk, free. Cross-layer shared state in
        # compressed attention (shared_attn.candidates/compress_kv/index_k) is published
        # by the kv_source layer and consumed by its followers, so it must be cleared at
        # every layer boundary in layer-major order (chunk-major keeps them within one
        # layer's chunk sequence, which is why it never needed this).
        self._timings = []
        for i in layer_seq:
            bobj = self.block(i)
            streamed = i in self._stream_layers()
            if streamed and self.device != "cpu":
                bobj.to(self.device)
            elif self.device != "cpu":
                bobj.to(self.device)
            for slot in acts:
                h_cpu, pre_cpu, hash_ids, off = slot
                h = h_cpu.to(self.device, non_blocking=False)
                pre_mix = pre_cpu.to(self.device, non_blocking=False)
                if bobj.engram is not None and hash_ids is not None:
                    h = bobj.engram.forward(h, hash_ids[bobj.engram.hash_index])
                h, pre_mix = bobj.forward(h, off, pre_mix)
                slot[0] = h.cpu()
                slot[1] = pre_mix.cpu()
            if getattr(bobj.ffn, "_last_ids", None):
                self._last_routing[i] = bobj.ffn._last_ids
            if streamed and self.device != "cpu":
                bobj.to("cpu")
        # 3) final head on the last chunk
        h_cpu, pre_cpu, _, _ = acts[-1]
        h = h_cpu.to(self.device, non_blocking=False)
        pre_mix = pre_cpu.to(self.device, non_blocking=False)
        self._last_hidden = h
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
                             temperature: float = 0.0, eos_token_id=None):
        """MTP speculative decoding: draft with the DSpark stages, verify with one main forward.

        Draft chain (official DFlash contract, see mtp.py):
            target hiddens -> main_x -> draft attention seed
            golden token   -> noise-filled draft block -> stage0..2 -> forward_head logits
        Verification: the main model's argmax at each accepted position must equal the
        draft token; the first mismatch truncates the run. KV is snapshotted/rolled back
        so the main model state stays exact. Returns (tokens, accept_rate).
        """
        from .nn import sample_logits
        if getattr(self, "mtp", None) is None and not self.load_mtp():
            raise RuntimeError("MTP weights are not available (model-00044..46 not downloaded?)")
        draft_len = draft_len or min(self.cfg.dspark_block_size, 4)
        stages, s0 = self.mtp, self.mtp[0]
        ids = list(prompt_ids)
        out: list[int] = []
        accepted_total = drafted_total = 0
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            while len(out) < max_new_tokens:
                nid = int(sample_logits(logits[0, -1], temperature))
                out.append(nid)
                if eos_token_id is not None and nid == eos_token_id:
                    break
                if len(out) >= max_new_tokens:
                    break
                # advance the main model with nid; captures dspark target hiddens
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=len(ids))
                ids.append(nid)
                th = self._target_hidden
                if not th:
                    continue
                main_x = s0.main_hiddens([th[i] for i in sorted(th.keys())])
                pos = len(ids)  # draft positions start after the accepted token
                x = s0.forward_embed(torch.tensor([[nid]], device=self.device))
                pre = torch.zeros(1, x.size(1), self.cfg.hc_mult, device=self.device)
                pre[:, :, 0] = 1.0
                for st in stages:
                    x, pre = st.forward(x, pos, pre, main_x)
                dlogits, _conf = stages[-1].forward_head(x, pre, torch.tensor([[nid]], device=self.device))
                if dlogits is None:
                    continue
                drafts = [int(dlogits[0, i].argmax()) for i in range(min(draft_len - 1, dlogits.size(1)))]
                if not drafts:
                    continue
                # verify: one main forward over the draft block, compare argmax position-wise
                snap = self.snapshot()
                vlogits = self.forward(torch.tensor([drafts], device=self.device), start_pos=pos)
                self.restore(snap)
                preds = [logits[0, -1]] + [vlogits[0, t] for t in range(len(drafts) - 1)]
                accepted = 0
                for t, cand in enumerate(drafts):
                    if int(preds[t].argmax()) == cand:
                        accepted += 1
                    else:
                        break
                drafted_total += len(drafts)
                accepted_total += accepted
                if accepted:
                    out.extend(drafts[:accepted])
                    ids.extend(drafts[:accepted])
                    logits = self.forward(torch.tensor([drafts[:accepted]], device=self.device), start_pos=pos)
                if len(out) >= max_new_tokens:
                    break
                if not accepted and len(drafts) and len(out) < max_new_tokens:
                    # no progress from the draft block: fall back to a plain decode step
                    pass
        rate = accepted_total / drafted_total if drafted_total else 0.0
        return out[:max_new_tokens], rate

    def generate(self, prompt_ids, max_new_tokens: int = 16, temperature: float = 0.0, eos_token_id=None,
                 top_p: float = 1.0, top_k: int = 0):
        """Greedy/sampled generation loop (prefill once, then decode step by step)."""
        ids = list(prompt_ids)
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            if self.device != "cpu":
                # the prefill kernels (tilelang workspace for the prompt-length shape)
                # leave ~5GB of cached blocks behind; against an 8GB card that pushes
                # reserved past physical memory and every decode step then pages
                # (measured 11498MB reserved, 0.10 tok/s). Hand them back to the driver.
                torch.cuda.empty_cache()
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
