"""TinyModel: direct-from-HF blocks and a minimal transformer driver."""
from __future__ import annotations

import torch

from .attention import MLAWindowAttention
from .compressed_attention import MLACompressedAttention
from .config import TinyConfig
from .hc import HCBlockMixin
from .moe import MoE
from .nn import rms_norm
from .weights import ShardStore


class Block(HCBlockMixin):
    """One decoder layer: Hyper-Connection collapse -> norm -> attention -> HC expand -> ... FFN."""

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, expert_store=None, quant: bool = False,
                 hashes=None, cache_wo_a: bool = False, expert_topk: int = 0, rt=None):
        self.layer_id = layer_id
        self.cfg = cfg
        self._expert_store = expert_store
        self.quant = quant
        self.cache_wo_a = cache_wo_a
        self.engram = None
        self._device = "cpu"
        self.profile = False
        self._stream_cache = None
        self._stream_flat = None    # (pinned cpu flat, device flat) for streamed layers
        self._flat_offsets = None
        self._cpu_refs = None       # original CPU tensors, restored on to("cpu")
        if hashes is not None and layer_id in cfg.engram_layer_ids:
            shard = store.shard_of(f"layers.{layer_id}.engram.embed.weight")
            if (store.model_dir / shard).exists():
                from .engram import Engram
                self.engram = Engram(store, layer_id, cfg, hashes)
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        if cfg.compress_ratios[layer_id] == 0:
            self.attn = MLAWindowAttention(store, layer_id, cfg, quant=quant, cache_wo_a=cache_wo_a)
        else:
            self.attn = MLACompressedAttention(store, layer_id, cfg, quant=quant, rt=rt)
        self.ffn = MoE(store, layer_id, cfg, expert_store=expert_store, topk=expert_topk)
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
        for name in ("q_norm", "kv_norm", "sink", "freqs_cis"):
            out.append((a, name, getattr(a, name)))
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
        self._cpu_refs = [t for (_, _, t) in paths]

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
            if getattr(self, "_cpu_refs", None) is None:
                self._device = "cpu"
                return self
            for (owner, name, _), ref in zip(paths, self._cpu_refs):
                setattr(owner, name, ref)
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
                try:
                    return t.pin_memory()
                except Exception:
                    return t

            self._cpu_refs = [_pin(t) for (_, _, t) in paths]
        import os as _os
        from .cpu_kernels import slab

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
                for (owner, name, t), cpu, o, n in zip(paths, self._cpu_refs, offsets, nbytes):
                    st[o:o + n].copy_(cpu.reshape(-1).view(torch.uint8), non_blocking=False)
                gpu = st[:total].to(device, non_blocking=False)  # single pinned H2D
                for (owner, name, t), o, n in zip(paths, offsets, nbytes):
                    setattr(owner, name, gpu[o:o + n].view(t.dtype).view(t.shape))
            else:
                for (owner, name, _), cpu in zip(paths, self._cpu_refs):
                    setattr(owner, name, cpu.to(device, non_blocking=True))
        else:
            for (owner, name, _), cpu in zip(paths, self._cpu_refs):
                setattr(owner, name, cpu.to(device, non_blocking=True))
        if getattr(self, "engram", None) is not None:
            # engram tables stay on disk (row reads) but its small projection weights
            # (q/k/wkv, cached bf16) must follow the block.
            self.engram.to(device)
        self._device = device
        return self

    def forward(self, x: torch.Tensor, start_pos: int, pre_mix: torch.Tensor):
        if self.profile:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            ev[0].record()
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.hc_pre(x, pre_mix)
        h = rms_norm(h, self.attn_norm, self.cfg.rms_norm_eps)
        # official DFlash dflash_input: the target layer's attn_norm output (attention
        # input, collapsed to [T, dim]); captured here for MTP stage-0 projection.
        self._last_attn_norm = h
        if self.profile:
            ev[1].record()
        h = self.attn.forward(h, start_pos)
        if self.profile:
            ev[2].record()
        x = self.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.hc_pre(x, attn_pre)
        h = rms_norm(h, self.ffn_norm, self.cfg.rms_norm_eps)
        if self.profile:
            ev[3].record()
        h = self.ffn.forward(h)
        if self.profile:
            ev[4].record()
        x = self.hc_post(h, residual, ffn_post, ffn_comb)
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

    def __init__(self, store: ShardStore, cfg: TinyConfig, layers: list[int] | None = None):
        self.store = store
        self.cfg = cfg
        self.embed = store.get("embed.weight")
        self.out_norm = store.get("norm.weight")
        self.head = store.get("head.weight")
        self.layers = layers or []
        self.blocks = [Block(store, i, cfg) for i in self.layers]

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
        for block in self.blocks:
            h, pre_mix = block.forward(h, start_pos, pre_mix)
        from .hc import hc_split_sinkhorn  # noqa: F401  (kept importable for users)
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.cfg.rms_norm_eps)
        # bf16 matvec (official head stores fp32 weights; bf16 is 2x faster on CPU and
        # numerically fine for sampling) -- see docs/loading.md head measurements
        logits = torch.nn.functional.linear(h, self.head).float()
        return logits
