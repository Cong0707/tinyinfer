"""TinyModel: direct-from-HF blocks and a minimal transformer driver."""
from __future__ import annotations

import torch

from .attention import MLAWindowAttention
from .compressed_attention import MLACompressedAttention
from ..config import TinyConfig
from .hc import HCBlockMixin, hc_pre_rms_norm

# Fused hc_pre + rms_norm: one kernel instead of two, 160 launches per decode token.
# Verified bit-identical across all 40 layers (first-difference probe over the whole
# stack), so it is the default; TI_FUSE=0 forces the original two-step pair for A/B.
#
# Getting it bit-identical required copying the unfused rms_norm's reduction structure
# (8-wide vectors, thread i owns vector i, same shuffle tree, same block size) rather than
# writing an equivalent reduction: one bf16 ulp per layer compounded into a different
# argmax by layer 5.
import os as _os_fuse

_FUSE_HC_PRE = _os_fuse.environ.get("TI_FUSE") != "0"
from .moe import MoE
from ..nn import rms_norm
from ..weights import ShardStore


def set_weight(owner, name: str, tensor) -> None:
    """Point a module's weight at a new tensor without losing its Parameter identity.

    Qwen's layers wrap their weights in nn.Parameter, and assigning a plain Tensor
    over one raises TypeError("cannot assign 'torch.BFloat16Tensor' as parameter
    'weight'"). That aborted the flat-streaming setup on Qwen3.8-27B and left every
    layer on the CPU, which surfaced far downstream as "mat2 is on cpu" in the head
    matmul. Writing through .data keeps the registered parameter and swaps its storage.
    """
    cur = getattr(owner, name, None)
    if isinstance(cur, torch.nn.Parameter):
        cur.data = tensor
    else:
        setattr(owner, name, tensor)
    # QuantLinear carries a separate `parts` tuple describing the (device, weight, scale)
    # layout -- one entry per card. flat streaming swaps a layer's bytes through this
    # function and can only reach the attribute, never that tuple, so a streamed layer's
    # parts kept pointing at its construction-time host copy. The owner is the only thing
    # that can know how to reconcile them, and a sharded layer must NOT be touched here
    # (its parts are the multi-card placement, not a cache), so this is a hook rather than
    # logic: QuantLinear re-syncs a single-part tuple and ignores everything else.
    _sync = getattr(owner, "_sync_parts_from_weight", None)
    if _sync is not None:
        _sync()


class Block(HCBlockMixin):
    """One decoder layer: Hyper-Connection collapse -> norm -> attention -> HC expand -> ... FFN."""

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, expert_store=None, quant: bool = False,
                 tp=None,
                 hashes=None, cache_wo_a: bool = False, expert_topk: int = 0, rt=None,
                 cache_shared_bf16: bool = False, kv_quant: bool | None = None,
                 plan_fn=None):
        self.layer_id = layer_id
        self.cfg = cfg
        self._expert_store = expert_store
        self.quant = quant
        # KV precision is its own axis: tier.kv_bits decides whether the window cache
        # round-trips through fp8. It used to ride on 'quant' (the weight flag), so a
        # tier could not ask for quantised weights with a full-precision KV cache.
        self.kv_quant = quant if kv_quant is None else kv_quant
        self.cache_wo_a = cache_wo_a
        self.engram = None
        self._device = "cpu"
        self.profile = False
        self._stream_cache = None
        self._stream_flat = None    # (pinned cpu flat, device flat) for streamed layers
        self._flat_offsets = None
        self._cpu_refs = None       # original CPU tensors, restored on to("cpu")
        # 批量 H2D：逐 tensor copy_ 每次要吃 ~84us 的 CPU dispatch（每步 2849 次 = 239ms，
        # 而 GPU kernel 只有 ~91ms）。这里把同一层的几十个 (dst, src, n) 一次性用
        # cudaMemcpyAsync 入队，Python 侧只走一遍循环，dispatch 成本从"每个 tensor 一次"
        # 降到"每个 tensor 一次指针运算"。
        self._h2d_batch = self._make_h2d_batch()

    def _make_h2d_batch(self):
        """返回一个把 [(dst, src, nbytes)] 批量提交到当前流的函数；不可用则 None。"""
        try:
            import ctypes
            import torch
            _lib = None
            for _nm in ("libcudart.so", "libcudart.so.12", "libcudart.so.13",
                        "cudart64_12.dll", "cudart64_13.dll"):
                try:
                    _lib = ctypes.CDLL(_nm)
                    break
                except OSError:
                    continue
            if _lib is None:
                return None
            _fn = _lib.cudaMemcpyAsync
            _fn.restype = ctypes.c_int
            _fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_int, ctypes.c_void_p]
            _get_stream = _lib.cudaStreamDefault if hasattr(_lib, "cudaStreamDefault") else None
            _MEMCPY_H2D = 1
        except Exception:
            return None

        _keep = []

        def _submit(pairs):
            try:
                stream = torch.cuda.current_stream().cuda_stream
            except Exception:
                return False
            dp = [dst.data_ptr() for dst, _s, _n in pairs]
            if not dp or any(x == 0 for x in dp):
                return False
            for dst, src, n in pairs:
                # src 是 pinned host tensor（Block._cpu_refs 就是 pin 过的）
                rc = _fn(ctypes.c_void_p(dst.data_ptr()),
                         ctypes.c_void_p(src.data_ptr()),
                         ctypes.c_size_t(int(n)),
                         _MEMCPY_H2D,
                         ctypes.c_void_p(stream))
                if rc != 0:
                    _keep.clear()
                    return False
            _keep.clear()
            _keep.extend(pairs)
            return True

        return _submit
        if hashes is not None and layer_id in cfg.engram_layer_ids:
            shard = store.shard_of(f"layers.{layer_id}.engram.embed.weight")
            if (store.model_dir / shard).exists():
                from .engram import Engram
                self.engram = Engram(store, layer_id, cfg, hashes)
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        # KV precision is its own axis (tier.kv_bits), but it must default to the weight
        # flag so the resident-layer plan is unchanged for every tier that does not set
        # it explicitly. kv_quant is resolved in Engine and passed down.
        if cfg.compress_ratios[layer_id] == 0:
            self.attn = MLAWindowAttention(store, layer_id, cfg, quant=self.kv_quant,
                                           cache_wo_a=cache_wo_a, tp=tp, plan_fn=plan_fn)
        else:
            self.attn = MLACompressedAttention(store, layer_id, cfg, quant=self.kv_quant,
                                               rt=rt, tp=tp, plan_fn=plan_fn)
        # the bf16 expert cache exists for the dequantised (reference) path; when the
        # block keeps fp8/fp4 the fp4 serve path reads the blob views directly, so eight
        # bf16 copies per layer only cost VRAM (40 x 8 x 30MB = 9.6GB, which is what made
        # a 63-token prefill run the allocator past physical memory)
        self.ffn = MoE(store, layer_id, cfg, expert_store=expert_store, topk=expert_topk,
                       cache_shared_bf16=cache_shared_bf16,
                       max_bf16_experts=(0 if quant and not __import__("os").environ.get("TI_BF16_CACHE")
                                          else 8),
                       plan_fn=plan_fn)
        p = f"layers.{layer_id}."
        self.attn_norm = store.get(p + "attn_norm.weight")
        self.ffn_norm = store.get(p + "ffn_norm.weight")
        self.hc_attn_fn = store.get(p + "hc_attn_fn").float()
        self.hc_attn_base = store.get(p + "hc_attn_base").float()
        self.hc_attn_scale = store.get(p + "hc_attn_scale").float()
        self.hc_ffn_fn = store.get(p + "hc_ffn_fn").float()
        self.hc_ffn_base = store.get(p + "hc_ffn_base").float()
        self.hc_ffn_scale = store.get(p + "hc_ffn_scale").float()

    def _weight_paths(self):
        """(owner, attr, tensor) for every weight; runtime state (underscore) excluded."""
        out = []
        for name in ("attn_norm", "ffn_norm", "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                     "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            out.append((self, name, getattr(self, name)))
        a = self.attn
        # freqs_cis is a shared model constant (rope_cache), not a per-layer weight:
        # streaming it cost 256MB x 36 layers = 9.2GB per token of duplication
        for name in ("q_norm", "kv_norm", "sink"):
            out.append((a, name, getattr(a, name)))
        if getattr(a, "freqs_cis", None) is not None and a.freqs_cis.device.type != "cpu":
            pass  # already resident on the device
        if getattr(a, "wqkv", None) is not None:
            # the stacked q/kv projection owns the device copy; wq_a/wkv stay on the host
            ql = a.wqkv
            out.append((ql, "weight", ql.weight))
            out.append((ql, "scale", ql.scale))
            for name in ("wq_b", "wo_a", "wo_b"):
                ql = getattr(a, name)
                if hasattr(ql, "weight"):
                    out.append((ql, "weight", ql.weight))
                    out.append((ql, "scale", ql.scale))
                else:
                    out.append((a, name, ql))
        else:
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                ql = getattr(a, name)
                if hasattr(ql, "weight"):
                    out.append((ql, "weight", ql.weight))
                    out.append((ql, "scale", ql.scale))
                else:
                    out.append((a, name, ql))
        comp = getattr(a, "compressor", None)
        if comp is not None:
            for name in ("norm_w", "wkv", "wgate"):
                t = getattr(comp, name, None)
                if isinstance(t, torch.Tensor):
                    out.append((comp, name, t))
        idx = getattr(a, "indexer", None)
        if idx is not None:
            for name in ("wq_b", "weights_proj", "wk", "k_norm", "freqs_cis"):
                t = getattr(idx, name, None)
                if isinstance(t, torch.Tensor):
                    out.append((idx, name, t))
        f = self.ffn
        for name in ("weight", "bias", "bias_vl"):
            t = getattr(f.gate, name, None)
            if isinstance(t, torch.Tensor):
                out.append((f.gate, name, t))
        for name in ("w1", "w2", "w3"):
            t = getattr(f.shared, name, None)
            if hasattr(t, "weight"):
                out.append((t, "weight", t.weight))
                out.append((t, "scale", t.scale))
            elif isinstance(t, torch.Tensor):
                out.append((f.shared, name, t))
        return out

    @staticmethod
    def _wkey(owner, name):
        """Stable identity for a weight slot: the owner object plus the attribute name.

        The reference list used to be positional, which broke silently as soon as
        _weight_paths gained or lost an entry (dropping the shared RoPE table shifted
        every later slot by one and the per-tensor copies landed on the wrong tensors).
        """
        return (id(owner), name)

    def _ensure_flat(self, device):
        """Pack all weights into one aligned pinned flat buffer + one device flat buffer."""
        if getattr(self, "_stream_flat", None) is not None:
            return
        paths = self._weight_paths()
        refs = [t for (_, _, t) in paths]
        # pack grouped by element size; pad every tensor to 256B so tilelang kernels
        # see aligned pointers (misaligned flat views segfault in vectorized loads)
        order = sorted(range(len(refs)), key=lambda j: -refs[j].element_size())
        offsets: dict[int, tuple[int, int]] = {}
        off = 0
        for j in order:
            t = refs[j]
            nbytes = t.numel() * t.element_size()
            off = (off + 255) // 256 * 256
            offsets[j] = (off, nbytes)
            off += nbytes
        total = off
        cpu_flat = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        for j, t in enumerate(refs):
            o, n = offsets[j]
            cpu_flat[o:o + n].copy_(t.reshape(-1).view(torch.uint8))
        gpu_flat = torch.empty(total, dtype=torch.uint8, device=device)
        self._stream_flat = (cpu_flat, gpu_flat)
        self._flat_offsets = offsets
        self._flat_order = order
        self._cpu_refs = {self._wkey(o, n): t for (o, n, t) in paths}

    def to(self, device):
        """Move/preload this block's weights.

        Weights are read-only at inference time: to CPU only swaps the original
        tensors back (zero copy), to CUDA does a single H2D of the packed flat
        buffer and points the attributes at views of it.
        """
        if device == self._device:
            return self
        paths = self._weight_paths()
        if device == "cpu":
            refs = getattr(self, "_cpu_refs", None)
            if refs is None:
                self._device = "cpu"
                return self
            for (owner, name, _) in paths:
                ref = refs.get(self._wkey(owner, name))
                if ref is not None:
                    set_weight(owner, name, ref)
            self._device = "cpu"
            return self
        # per-tensor copies: a packed flat buffer cannot be freed per layer (views pin
        # it), which leaked ~190MB/layer on 8GB cards
        if getattr(self, "_cpu_refs", None) is None:
            # pinned host memory makes the H2D run at pinned bandwidth. Windows can
            # refuse registrations once the per-process pinned pool is large
            # ("resource already mapped"); fall back to pageable memory in that case.
            def _pin(t):
                if not (t.is_cpu and t.device.type == "cpu"):
                    return t
                if t.is_pinned():
                    # already a pinned pool slice: pin_memory() would COPY it (another
                    # ~221MB per layer outside the pool), which must never happen once
                    # the pool owns the host footprint
                    return t
                try:
                    return t.pin_memory()
                except Exception:
                    return t

            self._cpu_refs = {self._wkey(o, n): _pin(t) for (o, n, t) in paths}
        import os as _os
        from ..cpu_kernels import slab

        st = None if _os.environ.get("TI_SLAB") is None else slab()
        if st is not None:
            # one layer = one pinned memcpy + one pinned H2D. Pageable per-tensor copies
            # were 763 transfers x 781us (8.3GB/s); a 190MB pinned H2D is ~7ms at 26GB/s.  # noqa: E501
            nbytes = [t.numel() * t.element_size() for (_, _, t) in paths]
            offsets = []
            off = 0
            for n in nbytes:
                off = (off + 255) // 256 * 256  # 256B alignment: tilelang kernels require it
                offsets.append(off)
                off += n
            total = (off + 255) // 256 * 256
            if total <= st.numel():
                for (owner, name, t), o, n in zip(paths, offsets, nbytes):
                    cpu = self._cpu_refs[self._wkey(owner, name)]
                    st[o:o + n].copy_(cpu.reshape(-1).view(torch.uint8), non_blocking=False)
                gpu = st[:total].to(device, non_blocking=False)  # single pinned H2D
                for (owner, name, t), o, n in zip(paths, offsets, nbytes):
                    set_weight(owner, name, gpu[o:o + n].view(t.dtype).view(t.shape))
            else:
                for (owner, name, _) in paths:
                    set_weight(owner, name,
                            self._cpu_refs[self._wkey(owner, name)].to(device, non_blocking=True))
        else:
            for (owner, name, _) in paths:
                set_weight(owner, name,
                        self._cpu_refs[self._wkey(owner, name)].to(device, non_blocking=True))
        if getattr(self, "engram", None) is not None:
            # engram tables stay on disk (row reads) but its small projection weights
            # (q/k/wkv, cached bf16) must follow the block.
            self.engram.to(device)
        self._device = device
        return self

    def forward(self, x: torch.Tensor, start_pos: int, pre_mix: torch.Tensor,
                positions=None):
        # Device alignment. With weights placed across cards (engine._gpu_for_layer) the
        # incoming activation still lives on the previous layer's card, so it has to move
        # before any of this layer's kernels run. On a single GPU the comparison is
        # torch.device("cuda") == cuda:0, i.e. always equal, so nothing is copied and the
        # path is identical to before. non_blocking keeps the copy async; the first
        # consumer is a kernel on the same stream, so ordering is already guaranteed.
        _dev = getattr(self, "_device", "cpu")
        if _dev != "cpu":
            _tdev = torch.device(_dev)
            if x.device != _tdev:
                x = x.to(_tdev, non_blocking=True)
            if pre_mix is not None and pre_mix.device != _tdev:
                pre_mix = pre_mix.to(_tdev, non_blocking=True)
        import os as _os
        _wt = _os.environ.get("TI_WALL_PROF") == "1"
        if _wt:
            import time as _t
            _w = [_t.perf_counter()]
        if self.profile:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            ev[0].record()
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        if _FUSE_HC_PRE:
            h = hc_pre_rms_norm(x, pre_mix, self.attn_norm, self.cfg.rms_norm_eps)
        else:
            h = self.hc_pre(x, pre_mix)
            h = rms_norm(h, self.attn_norm, self.cfg.rms_norm_eps)
        self._last_attn_norm = h
        if _wt:
            _w.append(_t.perf_counter())          # 1: attn input ready
        if self.profile:
            ev[1].record()
        h = self.attn.forward(h, start_pos, positions=positions)
        _dp = getattr(self, "_dump_parts", None)
        if _dp is not None:
            _dp.append((self.layer_id, "y_attn", h.clone()))
        if _wt:
            _w.append(_t.perf_counter())          # 2: attn submitted
        if self.profile:
            ev[2].record()
        x = self.hc_post(h, residual, attn_post, attn_comb)
        if _wt:
            _w.append(_t.perf_counter())          # 3: hc_post #1

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        if _FUSE_HC_PRE:
            h = hc_pre_rms_norm(x, attn_pre, self.ffn_norm, self.cfg.rms_norm_eps)
        else:
            h = self.hc_pre(x, attn_pre)
            h = rms_norm(h, self.ffn_norm, self.cfg.rms_norm_eps)
        if _wt:
            _w.append(_t.perf_counter())          # 4: ffn input ready
        if self.profile:
            ev[3].record()
        h = self.ffn.forward(h)
        if _dp is not None:
            _dp.append((self.layer_id, "y_ffn", h.clone()))
        if _wt:
            _w.append(_t.perf_counter())          # 5: MoE done
        if self.profile:
            ev[4].record()
        if _wt and not getattr(self, "_dev_dbg", False):
            self._dev_dbg = True
            print(f"[block {self.layer_id}] x={x.device} h={h.device} post={ffn_post.device} "
                  f"h_dtype={h.dtype} h_shape={tuple(h.shape)}", flush=True)
        x = self.hc_post(h, residual, ffn_post, ffn_comb)
        if _wt:
            _w.append(_t.perf_counter())          # 6: hc_post #2
            keys = ("attn_pre", "attn", "hc1", "ffn_pre", "moe", "hc2")
            parts = getattr(self, "_wall_parts", {k: 0.0 for k in keys})
            for i, k in enumerate(keys):
                parts[k] += (_w[i + 1] - _w[i]) * 1e3
            self._wall_parts = parts
        if self.profile:
            ev[5].record()
            torch.cuda.synchronize()
            self._parts = {
                "hc_pre": ev[0].elapsed_time(ev[1]),
                "attn": ev[1].elapsed_time(ev[2]),
                "hc_mid": ev[2].elapsed_time(ev[3]),
                "ffn": ev[3].elapsed_time(ev[4]),
                "hc_post": ev[4].elapsed_time(ev[5]),
            }
        return x, ffn_pre


class TinyModel:
    """Minimal driver: embed -> (selected blocks) -> norm -> head.

    Loading all 40 blocks would need ~500GB of resident weights, so this driver
    takes an explicit layer list for staged verification.
    """

    def __init__(self, store: ShardStore, cfg: TinyConfig, layers: list[int] | None = None,
                 tokenizer=None):
        self.store = store
        self.cfg = cfg
        self.embed = store.get("embed.weight")
        self.out_norm = store.get("norm.weight")
        self.head = store.get("head.weight")
        self.layers = layers or []
        # engram needs the compressed-token hash state; without the tokenizer the engram
        # layers would silently contribute nothing (which disagreed with the engine path)
        self.hashes = None
        if tokenizer is not None and cfg.engram_layer_ids:
            from .engram import EngramHashes

            self.hashes = EngramHashes(cfg, tokenizer)
        self.blocks = [Block(store, i, cfg, hashes=self.hashes) for i in self.layers]

    def reset(self) -> None:
        for block in self.blocks:
            if hasattr(block.attn, "reset"):
                block.attn.reset()
            block.ffn._expert_cache.clear()

    def forward(self, ids: torch.Tensor, start_pos: int = 0):
        h = self.embed[ids]  # [b, s, dim]
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        hashes = None
        if self.hashes is not None:
            hashes = self.hashes(ids.reshape(-1).tolist(), start_pos)
        for block in self.blocks:
            if block.engram is not None and hashes is not None:
                h = block.engram.forward(h, hashes[block.engram.hash_index])
            h, pre_mix = block.forward(h, start_pos, pre_mix)
        from .hc import hc_split_sinkhorn  # noqa: F401  (kept importable for users)
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.cfg.rms_norm_eps)
        # the official ParallelHead projects in fp32; chunk the vocab axis so the fp32
        # copy of the 129k x 5120 head never exists whole (2.65GB per call)
        h = h.float()
        head = self.head
        if head.dtype == torch.float32:
            return torch.nn.functional.linear(h, head)
        step = 32768
        out = None
        for start in range(0, head.shape[0], step):
            chunk = torch.nn.functional.linear(h, head[start:start + step].float())
            if out is None:
                out = torch.empty(*chunk.shape[:-1], head.shape[0],
                                  dtype=chunk.dtype, device=chunk.device)
            out[..., start:start + step] = chunk
        return out