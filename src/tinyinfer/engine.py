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
        # TI_TIER lets a caller pick the optimisation level without touching code
        import os as _os

        if "tier" not in kwargs and _os.environ.get("TI_TIER"):
            kwargs["tier"] = OptLevel(int(_os.environ["TI_TIER"]))
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
        tier: OptLevel | TierConfig = OptLevel.AGGRESSIVE,
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
        # tier first: the hardware plan reserves MTP when the tier enables speculation
        if isinstance(tier, str):
            from .tiers import plan as _tier_plan
            tier = _tier_plan(OptLevel[tier.upper()])
        self.tier = plan(tier) if isinstance(tier, OptLevel) else tier
        _topk = int(__import__("os").environ.get("TI_TOPK", "0"))
        if _topk > 0:
            self.tier.expert_topk = _topk
        # The streaming ring, the resident layers and the attention temporaries are
        # allocated and freed at very different lifetimes; on a card this close to full
        # the allocator's default segmenting leaves 0 free bytes even when the total is
        # under capacity. expandable segments lets PyTorch unmap rather than fragment.
        import os as _os2
        _os2.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # hardware-derived plan (VRAM/RAM budget -> resident layers, slots, cache, chunk)
        from .resources import model_profile as _model_profile
        from .resources import plan as _hw_plan
        if device != "cpu":
            # measure the checkpoint: per-layer / expert / embed / head bytes come from
            # the safetensors headers, so the solver multiplies real numbers instead of
            # a table tuned for one model on one card
            try:
                from .loading import ShardIndex

                _prof = _model_profile(ShardIndex(self.store.model_dir))
            except Exception:
                _prof = None
            # speculation reserves its draft stages up front (tier 2 is the default)
            self.hw_plan = _hw_plan(None, self.cfg.n_layers, model=_prof,
                                    speculation=bool(getattr(self.tier, "mtp", False)))
        else:
            self.hw_plan = None
        if self.hw_plan is not None:
            print(f"[tinyinfer] {self.hw_plan.reason}", flush=True)
            if not getattr(self.hw_plan, "fits", True) and self.device != "cpu":
                # Refuse loudly instead of OOMing after the first prefill chunk. The
                # streaming skeleton alone (ring + workspace + tables) does not fit.
                raise RuntimeError(
                    f"not enough VRAM for this checkpoint: {self.hw_plan.reason}. "
                    "The engine needs a card that can hold at least the streaming ring "
                    "plus the tilelang workspace; see docs/PERFORMANCE.md.")
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
                _slots = int(__import__("os").environ.get(
                    "TI_SLOTS", hw.expert_slots if hw is not None else 96))
                from .budget import budget as _host_budget
                _hb_room = _host_budget().room()
                _plan_cache = (hw.cache_bytes if hw is not None
                               else min(16 << 30, max(512 << 20, cache_budget)))
                # leave the majority of what is left to the pinned layer slab, which
                # _flat_setup requests later: 45% here, the rest there
                # The layer slab asks for its bytes AFTER this, so the cache may only
                # take a minority of what is free or the box starts paging to C: and
                # every read turns into a swap-in (observed: 9.3s/token).
                _cache_bytes = max(192 << 20, min(_plan_cache, int(_hb_room * 0.35)))
                self.expert_store = ExpertStore.from_store(
                    self.store,
                    slot_arena=bool(getattr(self.tier, "slot_arena", False)),
                    # parallel reads saturate the device around QD=8-16; scale with the
                    # core count instead of pinning a machine-specific thread count
                    reader_threads=min(16, max(4, (__import__("os").cpu_count() or 8))),
                    cache_bytes=_cache_bytes,
                    policy=__import__("os").environ.get("TI_CACHE_POLICY", "arc"),
                    vram_slots=_slots if device != "cpu" else 0,
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
            # hc_post on the host costs one D2H sync per call (four per layer). At M=1
            # that is cheap because the GPU queue is short; at M>1 the sync waits for
            # the whole layer (measured: 4 x 150ms x 38 layers = 24s of a 29s forward).
            self.small_ops_on_cpu = (small_op_on_cpu(self.gpu_layers, self.max_layers)
                                     and __import__("os").environ.get("TI_HC_GPU") != "1")
        if self.device != "cpu" and self.tier.level < OptLevel.AGGRESSIVE:
            # Reference-project behaviour: put the whole model on the device and let the
            # allocator decide. No streaming, no partial residency, no refusal -- if it
            # does not fit, the OOM is the answer (the same one vLLM would give).
            self.gpu_layers = self.max_layers
            print(f"[tinyinfer] tier {self.tier.level.name}: all {self.max_layers} layers "
                  f"resident (no streaming)", flush=True)
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
        # The embedding table is 1.32GB of VRAM for a 10KB row lookup per token. On a
        # card that cannot hold every layer it is pure waste: keeping it in host RAM
        # frees enough for two resident layers plus the expert staging pool.
        # TI_EMBED_GPU=1 forces the old behaviour (table in VRAM); it is the escape hatch
        # when a new heuristic regresses a machine that used to run.
        # Tensor parallel: one process holds every rank as a thread in Flare; here the
        # caller launches one process per rank and we join the default process group.
        self.tp = None
        self.tp_ranks = int(getattr(self.tier, "tp_ranks", 1) or 1)
        if self.tp_ranks > 1 and device != "cpu":
            import os as _os

            import torch.distributed as _dist

            if not (_dist.is_available() and _dist.is_initialized()):
                _dist.init_process_group(
                    backend="nccl",
                    rank=int(_os.environ.get("RANK", "0")),
                    world_size=self.tp_ranks,
                )
            from .tp import Shard

            self.tp = Shard(_dist.get_rank(), self.tp_ranks)
            print(f"[tinyinfer] tensor parallel rank {self.tp.rank}/{self.tp.world}", flush=True)
        self.embed_on_cpu = bool(getattr(self.hw_plan, "embed_on_cpu", False)) or device == "cpu"
        if __import__("os").environ.get("TI_EMBED_GPU") == "1":
            self.embed_on_cpu = False
        _emb = self.store.get("embed.weight")
        self.embed = _emb if self.embed_on_cpu else _emb.to(device)
        self.out_norm = self.store.get("norm.weight").to(device)
        # the 1.3GB output head: off-VRAM it costs ~40ms/token but frees 1.3GB
        self.head_on_cpu = device != "cpu" and not self.head_gpu
        self.head = self.store.get("head.weight") if self.head_on_cpu else self.store.get("head.weight").to(device)
        self.eps = self.cfg.rms_norm_eps
        # Tier >= 2 serves with speculation: load the DSpark stages now so the first
        # request does not pay for them. A missing/incompatible set disables MTP for
        # the session instead of failing the engine.
        _mtp_ok = bool(getattr(self.tier, "mtp", False)) and bool(
            getattr(self.hw_plan, "speculation", True))
        if __import__("os").environ.get("TI_MTP") == "1":
            # force the draft stages on even when the plan says they do not fit: used to
            # measure the accept rate on a card the planner keeps them off for
            _mtp_ok = True
        if getattr(self.tier, "mtp", False) and not _mtp_ok:
            print("[tinyinfer] MTP off: the draft stages do not fit beside the streaming "
                  "ring on this card (see the plan above)", flush=True)
        if _mtp_ok and device != "cpu":
            try:
                if self.load_mtp():
                    print("[tinyinfer] MTP draft stages loaded (tier %s)" % self.tier.level.name,
                          flush=True)
            except Exception as exc:
                print(f"[tinyinfer] MTP unavailable: {exc}", flush=True)

    # -- residency ---------------------------------------------------------
    def available_layers(self) -> int:
        """How many consecutive layers have all their shards fully on disk (size-checked).

        A safetensors header is readable long before the data segments finish
        downloading; loading such a shard segfaults inside CUDA. Compare the file
        size against header + max data offset instead.
        """
        cached = getattr(self, "_avail_layers", None)
        if cached is not None:
            return cached
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
        # Parsing every shard header costs ~10ms each and this is called on every
        # forward: 40 headers = 0.35s per decode step (cProfile). The set of layers on
        # disk does not change during a run.
        self._avail_layers = n
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
                      rt=self._rt_for(layer_id), tp=getattr(self, "tp", None))
            b.hc_post_on_cpu = getattr(self, "small_ops_on_cpu", False)
            ffn = getattr(b, "ffn", None)
            if ffn is not None:
                ffn._async_fetch = bool(getattr(self.tier, "async_expert_fetch", False))
                ffn._batched_scatter = bool(getattr(self.tier, "batched_scatter", False))
                if self.tp is not None and self.tp.world > 1:
                    ffn.ep_world = self.tp.world
                    ffn.ep_rank = self.tp.rank
            if layer_id < self.gpu_layers and self.device != "cpu":
                b.to(self.device)
            self._blocks[layer_id] = b
        return b

    def _layer_device(self, b: Block) -> str:
        return getattr(b, "_device", "cpu")

    def reset(self) -> None:
        self._last_routing.clear()
        # streaming completion events are keyed by layer and consumed by _flat_wait; a
        # stale entry makes the next forward rank a buffer copy that has not landed yet
        if hasattr(self, "_flat_ev"):
            self._flat_ev.clear()
        if self.hashes is not None:
            self.hashes.reset()
        for b in self._blocks.values():
            if hasattr(b.attn, "reset"):
                b.attn.reset()
            # the compressor carries the tail of an incomplete group across chunks; if
            # it is not cleared a second run from position 0 starts with stale scores
            comp = getattr(b.attn, "compressor", None)
            if comp is not None and hasattr(comp, "reset"):
                comp.reset()
            idx = getattr(b.attn, "indexer", None)
            if idx is not None and hasattr(idx, "reset"):
                idx.reset()
            b.ffn._expert_cache.clear()
        # Streaming state is per-forward. Dropping it alone would leave every streamed
        # block's weights pointing into a ring buffer that no longer exists, so restore
        # the CPU references first and let the next forward rebuild the ring.
        # per-forward caches on the shared handoff slot must not leak between sessions
        try:
            from .compressor import shared_attn as _sa0
            for _attr in ("_mask_cache", "candidates_by_chunk", "candidates"):
                if hasattr(_sa0, _attr):
                    try:
                        getattr(_sa0, _attr).clear()
                    except AttributeError:
                        setattr(_sa0, _attr, None)
        except Exception:
            pass
        # Streaming state is per-session. Re-point the streamed layers at their CPU
        # sources first (so no block holds a view into the ring), then release the pinned
        # host grant and drop the state: a rebuild that kept the old grant ran out of
        # budget and produced a partially initialised slab (silent numeric corruption).
        stream = self._stream_layers()
        for lid, b in self._blocks.items():
            if lid in stream:
                try:
                    b.to("cpu")
                except Exception:
                    pass
        # The slab/ring is a process-lifetime structure: rebuilding it per session leaks
        # the pinned grant and lets the caching allocator accumulate the old GPU buffers
        # (measured 17GB allocated on an 8GB card after three sessions). Keep it; only
        # the completion events belong to a single forward.
        if hasattr(self, "_flat_ev"):
            self._flat_ev.clear()
        if hasattr(self, "_flat_ev"):
            self._flat_ev.clear()
        if hasattr(self, "_flat_applies"):
            self._flat_applies = 0
        # inter-layer handoff slots: produced by a kv_source layer and consumed by the
        # layers after it inside one forward. A fresh run must not inherit the previous
        # run's compressed KV / index-K, which is what produced NaNs on a repeat prefill.
        try:
            from .compressor import shared_attn as _sa
            _sa.reset()
        except Exception:
            pass

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
            import os as _os

            flag = _os.environ.get("TI_FLAT")
            if flag is not None and flag != "1":
                self._flat_state = "off"
                return False
            if flag is None and self.gpu_layers >= self.max_layers:
                # every layer is resident: there is nothing to stream
                self._flat_state = "off"
                return False
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
        # ring depth is measurable: 3 hides one H2D behind compute, 2 frees a layer-sized
        # buffer (719MB here) for one more resident layer on a card this tight
        _ring_n = max(2, min(4, int(__import__("os").environ.get("TI_RING", "3"))))
        bufs = [torch.empty(mx, dtype=torch.uint8, device=self.device) for _ in range(_ring_n)]
        pinned_gb = 0.0
        # Pinned budget. Pinning all 17.9GB of streamed weights starved the expert
        # cache: pinned sources are the only ones that can use async DMA, and the
        # pinned pool also has to hold the expert blobs.
        from .budget import budget as _host_budget

        _hb = _host_budget()
        # Per-tensor fallback only: pin as much as the budget has left, one tensor at
        # a time, and let the ones that do not fit stay pageable.
        pin_budget = int(float(__import__("os").environ.get("TI_FLAT_PIN_GB", "0")) * (1 << 30))
        if pin_budget <= 0:
            pin_budget = min(_hb.room(), int(_hb.total * 0.36))
        # One pinned slab for every streamed layer, laid out exactly like the device
        # buffers, so a layer costs ONE flat H2D instead of 15-30 small copy_ calls.
        # cProfile: the per-tensor path issued 1469 copy_ calls and burned 1.01s of a
        # 3.45s decode step in host-side dispatch, while the pinned bytes were already
        # resident (the slab replaces them, it does not add to them).
        slab_total = sum(sizes[i] for i in streamed) + 512
        # Pinning every streamed layer is not the goal; it is only worth what it saves.
        # On a box where the weights alone are half of RAM, pin at most half of the
        # remaining room so the page cache (and the experts) still have memory.
        # Cap the pinned set at a share of PHYSICAL RAM, not of the volatile "room":
        # the slab plus the expert cache plus the 17GB of streamed weights must leave
        # the page cache alive, or the SSD drops from 4.6 to 3.1GB/s and every fetch
        # doubles (measured: full-RAM slab -> 2.29s fetch, capped slab -> 1.47s).
        # Fixed share of PHYSICAL RAM, not of the volatile room: the room shrinks as the
        # model weights load, so a room-derived cap collapsed to 6.7GB and left 21 of 35
        # layers on the slow per-tensor path. The invariant that matters is
        # weights(17GB) + slab + expert cache <= RAM - OS, and the planner already sized
        # the cache from the same budget, so the slab takes its share of the total.
        _slab_cap = int(_hb.total * 0.28)
        if slab_total > _slab_cap:
            slab_total = _slab_cap
        host_slab = None
        # Ask the host budget, which re-reads what the machine actually has left right
        # now. No flag, no per-machine constant: on a small box the grant comes back
        # short and we fall back to per-tensor pinned copies, on a big one we get the
        # whole slab. The slab REPLACES those copies, so the freed memory is available
        # to the next request automatically (available() is re-read every time).
        # One pinned block PER LAYER rather than one giant block: a single 15GB
        # cudaHostAlloc is refused by the driver on some Windows configurations (the
        # per-tensor fallback below pinned the same 14.9GB in small pieces), while a
        # 468MB block always succeeds. Each block is still laid out exactly like the
        # device buffer, so a layer is one flat H2D, which is the whole point.
        host_map: dict[int, torch.Tensor] = {}
        _why = "ok"
        # One layer at a time: allocate its block, copy the weights in, point the layer
        # at the block, then let the old pageable copy go. Allocating all blocks first
        # holds both copies at once (+18GB) and that is what pushed this box into swap.
        _slab_used = 0
        for i in streamed:
            need = sizes[i]
            if _slab_used + need > slab_total:
                _why = (f"reached the pinned cap {slab_total/2**30:.1f}GB "
                        f"({len(host_map)}/{len(streamed)} layers)")
                break
            got = _hb.grant(need)
            if got < need:
                _why = f"budget: wanted {need/2**20:.0f}MB got {got/2**20:.0f}MB ({_hb.stats()})"
                break
            _slab_used += need
            self._slab_granted = getattr(self, "_slab_granted", 0) + need
            try:
                blk = torch.empty(need, dtype=torch.uint8, pin_memory=True)
            except Exception as exc:
                _hb.release(need)
                _why = f"pinned alloc failed at layer {i}: {exc!r}"
                break
            b = self.block(i)
            paths = b._weight_paths()
            offs = lay[i][0]
            for (owner, name, t), (o, n) in zip(paths, offs):
                dst = blk[o:o + n].view(t.dtype).view(t.shape)
                try:
                    dst.copy_(t)
                except Exception:
                    pass
                setattr(owner, name, dst)
            _refs = {}
            for (owner, name, t), (o, n) in zip(paths, offs):
                _refs[Block._wkey(owner, name)] = blk[o:o + n].view(t.dtype).view(t.shape)
            b._cpu_refs = _refs
            b._device = "cpu"
            host_map[i] = blk
            del paths, offs
            keep.append(i)
        if len(host_map) == len(streamed):
            pinned_gb = sum(int(v.numel()) for v in host_map.values()) / 2 ** 30
            self._flat_state = (lay, bufs, host_map)
            self._host_slab = host_map
        elif host_map:
            # Partial slab: the layers that got a block keep the one-copy fast path, the
            # rest fall back to per-tensor pinned copies. All-or-nothing wasted the RAM
            # that WAS available and left every layer on the slow path.
            print(f"[tinyinfer] partial slab {len(host_map)}/{len(streamed)} layers ({_why})",
                  flush=True)
            pinned_gb = sum(int(v.numel()) for v in host_map.values()) / 2 ** 30
            # the slab already consumed its share of the pinned budget; counting from
            # zero here pinned a second full budget and took the box to its page file
            pinned_used = int(pinned_gb * 2 ** 30)
            for i in streamed:
                if i in host_map:
                    continue
                b = self.block(i)
                paths = b._weight_paths()
                refs = []
                for (_, _, t) in paths:
                    nbytes = t.numel() * t.element_size()
                    if (t.is_cpu and not t.is_pinned() and pinned_used + nbytes <= pin_budget):
                        try:
                            t = t.pin_memory()
                            pinned_used += nbytes
                        except Exception:
                            pass
                    refs.append(t)
                # keyed the same way Block.to() keys its map, so the two stay in sync
                b._cpu_refs = {Block._wkey(o, n): t for (o, n, _), t in zip(paths, refs)}
                for (owner, name, _), t in zip(paths, refs):
                    setattr(owner, name, t)
                b._device = "cpu"
                pinned_gb += sum(t.numel() * t.element_size() for t in refs if t.is_pinned()) / 2 ** 30
                keep.append(i)
            self._flat_state = (lay, bufs, host_map)
            self._host_slab = host_map
        else:
            print(f"[tinyinfer] slab unavailable ({_why}); falling back to per-tensor pinned",
                  flush=True)
            pinned_used = 0
            for i in streamed:
                b = self.block(i)
                paths = b._weight_paths()
                refs = []
                for (_, _, t) in paths:
                    nbytes = t.numel() * t.element_size()
                    if (t.is_cpu and not t.is_pinned() and pinned_used + nbytes <= pin_budget):
                        try:
                            t = t.pin_memory()
                            pinned_used += nbytes
                        except Exception:
                            pass
                    refs.append(t)
                b._cpu_refs = {Block._wkey(o, n): t for (o, n, _), t in zip(paths, refs)}
                # repoint the layer at the pinned copies so the pageable originals can be
                # freed: keeping both meant 15.9GB + 15.9GB resident, and under that pressure
                # the pinned H2D fell from 30GB/s to 6.5GB/s (measured T=1 vs T=2)
                for (owner, name, _), t in zip(paths, refs):
                    setattr(owner, name, t)
                b._device = "cpu"
                pinned_gb += sum(t.numel() * t.element_size() for t in refs if t.is_pinned()) / 2 ** 30
                keep.append(i)
            self._flat_state = (lay, bufs, None)
        self._flat_ev = {}
        self._copy_stream = torch.cuda.Stream(device=self.device)
        print(f"[tinyinfer] flat streaming: {len(keep)} layers, {mx/2**20:.0f}MBx3 VRAM, "
              f"{pinned_gb:.1f}GB {'slab' if host_slab is not None else 'per-tensor'} pinned source",
              flush=True)

    def _flat_apply(self, layer: int, buf_idx: int) -> None:
        self._flat_applies = getattr(self, "_flat_applies", 0) + 1
        lay, bufs, host = self._flat_state
        b = self.block(layer)
        offs, total = lay[layer]
        buf = bufs[buf_idx]
        if __import__("os").environ.get("TI_DEBUG_FLAT"):
            print(f"[flat] layer {layer} buf {buf_idx} slab={host is not None and layer in host}",
                  flush=True)
        if host is None or layer not in host:
            # per-tensor copies from the pinned refs Block already holds (a partial
            # slab leaves some layers on this path). _cpu_refs is keyed by (owner, name)
            # now, not positional.
            for (owner, name, t), (o, n) in zip(b._weight_paths(), offs):
                src = b._cpu_refs[Block._wkey(owner, name)]
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
        _tt0 = __import__("time").perf_counter() if __import__("os").environ.get("TI_LAYER_TIME") else None
        h = self.embed[ids.cpu()].to(self.device) if self.embed_on_cpu else self.embed[ids]
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32, device=h.device)
        pre_mix[:, :, 0] = 1.0
        hash_ids = None
        if self.hashes is not None:
            _h0 = __import__("time").perf_counter() if _tt0 is not None else None
            hash_ids = self.hashes(ids.reshape(-1).tolist(), start_pos)
            if _h0 is not None:
                print(f"[time] pre-embed {( _h0 - _tt0)*1e3:.1f}ms hash {(__import__('time').perf_counter()-_h0)*1e3:.1f}ms", flush=True)
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
                    # ring size is dynamic: an OOM shed may have shrunk it, so wrap by
                    # the live length instead of a hard-coded 3
                    _ring = len(self._flat_state[1]) if isinstance(self._flat_state, tuple) else 1
                    _ring = max(1, _ring)
                    self._flat_wait(i, k % _ring)
                    for d in range(1, _ring):
                        if k + d < len(flat_seq):
                            nxt = flat_seq[k + d]
                            if nxt not in self._flat_ev:
                                self._flat_prefetch(nxt, (k + d) % _ring)
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
                _e0 = __import__("time").perf_counter() if __import__("os").environ.get("TI_LAYER_TIME") else None
                h = b.engram.forward(h, hash_ids[b.engram.hash_index])
                if _e0 is not None:
                    print(f"[time] layer {i} engram {(__import__('time').perf_counter()-_e0)*1e3:.1f}ms", flush=True)
            # official (reference/official/model.py:1267): the MTP head reads the *input* of
            # its target layers, hc-collapsed by a plain mean -- not the attn_norm output
            # and not the attention input. Captured before forward() overwrites h.
            _lt0 = __import__("time").perf_counter() if __import__("os").environ.get("TI_LAYER_TIME") else None
            if __import__("os").environ.get("TI_MEM_TRACE"):
                print(f"[mem] layer {i:2d} before alloc={torch.cuda.memory_allocated()/2**20:.0f}MB "
                      f"reserved={torch.cuda.memory_reserved()/2**20:.0f}MB", flush=True)
            h_in = h
            try:
                h, pre_mix = b.forward(h, start_pos, pre_mix)
            except Exception:
                print(f"[layer-fail] layer {i} streamed={streamed} use_flat={use_flat} "
                      f"device={getattr(b, '_device', '?')} "
                      f"hc_fn={getattr(b, 'hc_attn_fn', None) is not None and b.hc_attn_fn.device}",
                      flush=True)
                raise
            if __import__("os").environ.get("TI_DEBUG_NAN"):
                _fin = bool(torch.isfinite(h).all())
                if (not _fin) or __import__("os").environ.get("TI_DEBUG_NAN") == "all":
                    print(f"[dbg] layer {i:2d} finite={_fin} absmax={float(h.abs().max()):.2f}",
                          flush=True)
            if i in self._target_layers:
                self._target_hidden[i] = h_in.mean(dim=2).detach()
            if self.profile:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                events.append((i, ev0, ev1))
            if _lt0 is not None:
                _lt1 = __import__("time").perf_counter()
                print(f"[time] layer {i} total {(_lt1-_lt0)*1e3:.1f}ms "
                      f"streamed={streamed} use_flat={use_flat}", flush=True)
            if b.ffn._last_ids:
                self._last_routing[i] = b.ffn._last_ids
            if streamed and self.device != "cpu" and not use_flat:
                b.to("cpu")  # free VRAM for the next streamed layer
        if self.predictor is not None and tok_now is not None and self._last_routing:
            self.predictor.observe(tok_now, dict(self._last_routing))
        self._last_hidden = h  # [1, T, hc, dim] before collapse (MTP main hidden)
        # official DSparkBlock.forward: at start_pos == 0 a draft stage only *seeds* its
        # main-KV window (attn(x, 0, main_x)); without this the draft attends over an
        # empty window and its predictions are noise (measured accept 3% instead of ~50%)
        if start_pos == 0 and getattr(self, "mtp", None) is not None:
            try:
                _th = sorted(getattr(self, "_target_hidden", {}) or {})
                _want = list(self.cfg.dspark_target_layer_ids)
                if len(_th) >= len(_want):
                    _mx = self.mtp[0].main_hiddens([self._target_hidden[k] for k in _th])
                    for _st in self.mtp:
                        _seed = _mx.new_zeros(1, self.cfg.dspark_block_size, self.cfg.dim)
                        _st.attn.forward_with_main(_seed, 0, _mx)
            except Exception as _exc:
                print(f"[tinyinfer] MTP seed skipped: {_exc!r}", flush=True)
        self._last_pre_mix = pre_mix  # exact HC mixing used for the collapse
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
            h = self.embed[chunk_ids.cpu()].to(self.device) if self.embed_on_cpu else self.embed[chunk_ids]
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

    # -- VRAM pressure + CUDA graph + prefix checkpoints ---------------------
    def _shed_vram(self) -> None:
        """Give memory back after an OOM: shrink the streaming ring, drop the slot pool."""
        st = getattr(self, "_flat_state", None)
        if st is not None and st != "off":
            ring = list(st[1])
            if len(ring) > 2:
                # keep the two buffers decode needs and actually release the rest: a
                # shorter list alone does not give the VRAM back
                keep = ring[:2]
                dropped = ring[2:]
                self._flat_state = (st[0], keep, st[2])
                for buf in dropped:
                    del buf
                print(f"[tinyinfer] CUDA OOM: streaming ring 3 -> {len(keep)} buffers "
                      f"(released {len(dropped)})", flush=True)
        try:
            if getattr(self, "expert_store", None) is not None:
                self.expert_store.drop_slots()
        except Exception:
            pass
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def _graph_capture_shape(self, ids: torch.Tensor, start_pos: int) -> bool:
        if not getattr(self.tier, "cuda_graph", False):
            return False
        if self.device == "cpu" or ids.shape[1] != 1:
            return False
        if self.gpu_layers < self.max_layers:
            return False
        if getattr(self, "mtp", None) is not None:
            return False
        return True

    def _graph_run(self, ids: torch.Tensor, start_pos: int):
        g = getattr(self, "_graph", None)
        if g is None or start_pos != getattr(self, "_graph_start_pos", -1):
            return None
        out = g["out"]
        if out.shape[0] != ids.shape[0]:
            return None
        g["in"].copy_(ids)
        g["graph"].replay()
        return out.clone()

    def _graph_try_capture(self, ids: torch.Tensor, start_pos: int) -> None:
        if getattr(self, "_graph", None) is not None or getattr(self, "_graph_failed", False):
            return
        if not self._graph_capture_shape(ids, start_pos):
            return
        try:
            snap = self.snapshot()
            try:
                in_buf = ids.clone()
                self._forward_impl(in_buf, start_pos, None)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = self._forward_impl(in_buf, start_pos, None)
                torch.cuda.synchronize()
            finally:
                self.restore(snap)
            self._graph = {"graph": g, "in": in_buf, "out": out, "start_pos": start_pos}
            self._graph_start_pos = start_pos
            print(f"[tinyinfer] decode CUDA graph captured at pos={start_pos}", flush=True)
        except Exception as exc:
            self._graph_failed = True
            self._graph = None
            print(f"[tinyinfer] CUDA graph unavailable ({exc!r}); eager decode", flush=True)

    def warm_graph_catalog(self) -> None:
        caps = list(getattr(self.tier, "graph_catalog", ()) or ())
        if not caps or self.device == "cpu":
            return
        ids = torch.tensor([[self.cfg.bos_token_id or 0]], device=self.device)
        for cap in caps:
            try:
                self._forward_impl(ids.repeat(1, min(int(cap), 8)), 0, None)
            except Exception:
                continue
            break

    def save_checkpoint(self, path) -> dict:
        import os as _os

        if not getattr(self.tier, "prefix_checkpoint", False):
            raise RuntimeError("prefix checkpoints are a tier-2 mechanism "
                               f"(current tier {self.tier.level.name})")
        snap = self.snapshot()
        tmp = str(path) + ".tmp"
        torch.save(snap, tmp)
        _os.replace(tmp, str(path))
        return {"path": str(path), "bytes": _os.path.getsize(str(path))}

    def load_checkpoint(self, path) -> None:
        if not getattr(self.tier, "prefix_checkpoint", False):
            raise RuntimeError("prefix checkpoints are a tier-2 mechanism "
                               f"(current tier {self.tier.level.name})")
        self.restore(torch.load(str(path), map_location=self.device, weights_only=False))

    def resample_from_hidden(self, hidden=None, temperature: float = 0.0):
        from .nn import rms_norm as _rms
        from .nn import sample_logits as _sample

        if not getattr(self.tier, "resume_sampling", False):
            raise RuntimeError("resume sampling is a tier-2 mechanism "
                               f"(current tier {self.tier.level.name})")
        h = self._last_hidden if hidden is None else hidden
        if h is None:
            raise RuntimeError("no hidden state to sample from")
        if h.dim() == 4:
            pm = getattr(self, "_last_pre_mix", None)
            if pm is not None and pm.shape[:2] == h.shape[:2]:
                y = (pm.unsqueeze(-1) * h.float()).sum(dim=2)
            else:
                y = h.float().mean(dim=2)
        else:
            y = h.float()
        y = _rms(y.to(self.out_norm.dtype), self.out_norm, self.eps)
        if self.head_on_cpu:
            logits = torch.nn.functional.linear(y.cpu().to(self.head.dtype), self.head).float().to(y.device)
        else:
            logits = torch.nn.functional.linear(y.to(self.head.dtype), self.head).float()
        return int(_sample(logits[0, -1], temperature))

    def forward(self, ids: torch.Tensor, start_pos: int = 0, chunk: int | None = None):
        """Decode/prefill entry point: CUDA graph replay when captured, else the OOM ladder."""
        for attempt in range(3):
            try:
                if start_pos > 0 and chunk is None:
                    hit = self._graph_run(ids, start_pos)
                    if hit is not None:
                        return hit
                    self._graph_try_capture(ids, start_pos)
                return self._forward_impl(ids, start_pos, chunk)
            except Exception as exc:
                # torch raises OutOfMemoryError, while some CUDA paths surface a plain
                # RuntimeError whose text says "out of memory"; both are handled the same
                oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if attempt == 2 or not oom:
                    raise
                print(f"[tinyinfer] CUDA OOM: shedding VRAM (attempt {attempt + 1})", flush=True)
                self._shed_vram()
        raise RuntimeError("unreachable")

    def _forward_impl(self, ids: torch.Tensor, start_pos: int = 0, chunk: int | None = None):
        """Chunked prefill: KV/hash state carries across chunks (attention uses start_pos)."""
        if ids.device.type != self.device and self.device != "cpu":
            ids = ids.to(self.device)
        b, T = ids.shape
        step = chunk or self.prefill_chunk
        # Measured on this card: a 63-token prompt at chunk=8 runs the allocator past
        # physical VRAM (17GB "allocated" through WDDM shared memory) and OOMs, while the
        # same prompt at chunk=1 completes. Long prompts therefore step through in small
        # pieces; short ones keep the fast chunk. The growth is per-layer and linear in
        # the chunk length, so 16 is a safe bound below the blow-up.
        if chunk is None and T > 16:
            step = max(1, min(step, 16 // max(1, T // 16)))
            step = max(1, step)
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
    def _window_span(self) -> int:
        """Tokens still replayable from the ring without cloning it."""
        spans = [getattr(b.attn, "window", 0) or 0 for b in self._blocks.values()]
        return min([s for s in spans if s] or [0]) or 0

    def _kv_truncate(self, target: int) -> None:
        """Roll every attention state back to `target` absolute tokens."""
        for b in self._blocks.values():
            a = b.attn
            win = getattr(a, "window", None) or 0
            keep = target if not win else min(target, win)
            kv = getattr(a, "_kv_window", None)
            if kv is not None and kv.shape[1] > keep:
                a._kv_window = kv[:, :keep].contiguous()
            if hasattr(a, "_decode_win_idxs"):
                a._decode_win_idxs = None
            comp = getattr(a, "compressor", None)
            if comp is not None and comp.ratio > 1:
                want = target % comp.ratio
                for st in comp._states.values():
                    pk = st.get("pkv")
                    if pk is not None and pk.shape[1] > want:
                        st["pkv"] = pk[:, :want].contiguous()
                        sc = st.get("psc")
                        if sc is not None:
                            st["psc"] = sc[:, :want].contiguous()

    def _draft(self, nid: int, pos: int, draft_len: int) -> list:
        """One DSpark draft block: target hiddens -> seed -> stages -> head."""
        stages = self.mtp
        s0 = stages[0]
        th = self._target_hidden
        if not th:
            return []
        keys = sorted(th.keys())
        if len(keys) < len(self.cfg.dspark_target_layer_ids):
            return []
        # a draft seeds its attention from the main KV of the current token; state from
        # the previous draft must not leak in (measured: draft 2 vs draft 3 disagreed
        # on the 4th token because the window kept the earlier block)
        for st in stages:
            st.attn.reset()
        main_x = s0.main_hiddens([th[k] for k in keys])
        tok = torch.tensor([[nid]], device=self.device)
        x = s0.forward_embed(tok)
        pre = torch.zeros(1, x.size(1), self.cfg.hc_mult, device=self.device)
        pre[:, :, 0] = 1.0
        for st in stages:
            x, pre = st.forward(x, pos, pre, main_x)
        dlogits, _conf = stages[-1].forward_head(x, pre, tok)
        if dlogits is None:
            return []
        import os as _os
        if _os.environ.get("TI_SPEC_DEBUG"):
            top = dlogits[0, 0].topk(3)
            print(f"[draft] pos={pos} main_x_mean={float(main_x.abs().mean()):.4f} "
                  f"hidden_mean={float(x.abs().mean()):.3f} top3={[int(i) for i in top.indices]}",
                  flush=True)
        n = min(draft_len, dlogits.size(1))
        return [int(dlogits[0, i].argmax()) for i in range(n)]
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
        """MTP speculative decoding: one main forward verifies a whole draft block.

        Draft (official DFlash contract, see mtp.py):
            target hiddens of layers 37/38/39 -> main_proj/main_norm -> main_x
            golden token -> noise-filled draft block -> stage 0..2 -> head logits
        Verify:
            one main forward over the drafts; preds[t] is the argmax at each position;
            accept the longest matching prefix, then truncate the KV to pos+accepted so
            the rejected positions never enter the cache. The logits of the last
            accepted position become the next step's logits -- nothing is replayed.
        """
        from .nn import sample_logits

        if getattr(self, "mtp", None) is None and not self.load_mtp():
            raise RuntimeError("MTP weights are not available (model-00044..46 missing?)")
        draft_len = draft_len or self.cfg.dspark_block_size
        ids = list(prompt_ids)
        out: list[int] = []
        acc = drafted = 0
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            if self.device != "cpu":
                torch.cuda.empty_cache()
            pos = len(ids)
            while len(out) < max_new_tokens:
                nid = int(sample_logits(logits[0, -1], temperature))
                out.append(nid)
                if eos_token_id is not None and nid == eos_token_id:
                    break
                if len(out) >= max_new_tokens:
                    break
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=pos)
                pos += 1
                drafts = self._draft(nid, pos, draft_len)
                if not drafts:
                    continue
                span = self._window_span()
                cheap = span == 0 or pos + len(drafts) <= span
                snap = None if cheap else self.snapshot()
                vlogits = self.forward(torch.tensor([drafts], device=self.device), start_pos=pos)
                preds = [logits[0, -1]] + [vlogits[0, t] for t in range(len(drafts) - 1)]
                j = 0
                for t, cand in enumerate(drafts):
                    if int(preds[t].argmax()) == cand:
                        j += 1
                    else:
                        break
                import os as _os
                if _os.environ.get("TI_SPEC_DEBUG"):
                    _p0 = int(preds[0].argmax())
                    print(f"[spec] pos={pos} draft0={drafts[0]} pred0={_p0} "
                          f"match={_p0 == drafts[0]} logits_argmax={int(logits[0, -1].argmax())} "
                          f"n_drafts={len(drafts)}", flush=True)
                drafted += len(drafts)
                acc += j
                if j < len(drafts):
                    if cheap:
                        self._kv_truncate(pos + j)
                    else:
                        self.restore(snap)
                        if j:
                            vlogits = self.forward(
                                torch.tensor([drafts[:j]], device=self.device), start_pos=pos)
                if j:
                    out.extend(drafts[:j])
                    pos += j
                    logits = vlogits[0, j - 1:j]
                    if eos_token_id is not None and drafts[j - 1] == eos_token_id:
                        break
        rate = acc / drafted if drafted else 0.0
        return out[:max_new_tokens], rate


    def generate(self, prompt_ids, max_new_tokens: int = 16, temperature: float = 0.0, eos_token_id=None,
                 top_p: float = 1.0, top_k: int = 0):
        """Greedy/sampled generation loop (prefill once, then decode step by step)."""
        # a generation starts from a clean session: leftover KV from a previous call
        # makes the first step read stale state (observed as a first-token failure)
        self.reset()
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
            import time as _t
            _t0 = _t.perf_counter()
            for step in range(max_new_tokens):
                nid = int(sample_logits(logits[0, -1], temperature, top_p, top_k))
                out.append(nid)
                if eos_token_id is not None and nid == eos_token_id:
                    break
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=len(ids) + step)
                if __import__("os").environ.get("TI_TIME"):
                    _t1 = _t.perf_counter()
                    print(f"  decode {step} {_t1 - _t0:.2f}s", flush=True)
                    _t0 = _t1
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
