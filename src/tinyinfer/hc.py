"""Hyper-Connections: Sinkhorn-normalized mix coefficients, collapse/expand of hc copies."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .nn import rms_norm


# Fused-kernel availability is a process-wide fact. Latching it per Block meant 40
# instances each paid the failing compile/raise (~18ms) twice per token = 1.5s/token.
# OFF by default: the tilelang fused kernels reduce with atomics, so their results are
# not bit-reproducible; greedy sampling then diverges to different answers on identical
# input (observed: the same prompt answered "Paris" in one run and an unrelated essay in
# the next). The plain torch path is deterministic and is what "usable output" requires.
_FUSED_PRE_OK = False
_FUSED_POST_OK = False
_FUSED_MIXES_OK = False


def _sinkhorn_host(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    """Same maths, but the 2x2 Sinkhorn runs on the host.

    On this card the whole comb is 4 floats: 20 iterations of row/column normalisation
    meant ~10k device ops per token (sum/div launches of a 4-element kernel) costing
    ~70ms of host time and ~19ms of GPU time. A 4-float round trip plus 80 host multiplies
    is ~20 microseconds.
    """
    import numpy as np

    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    )
    raw = (mixes[..., 2 * hc_mult:].view(b, s, hc_mult, hc_mult) * hc_scale[2]
           + hc_base[2 * hc_mult:].view(hc_mult, hc_mult))
    c = raw.detach().float().cpu().numpy().astype(np.float32)
    e = np.exp(c - c.max(axis=-1, keepdims=True))
    c = (e / e.sum(axis=-1, keepdims=True) + np.float32(eps)).astype(np.float32)
    c = (c / (c.sum(axis=-2, keepdims=True) + np.float32(eps))).astype(np.float32)
    for _ in range(iters - 1):
        c = (c / (c.sum(axis=-1, keepdims=True) + np.float32(eps))).astype(np.float32)
        c = (c / (c.sum(axis=-2, keepdims=True) + np.float32(eps))).astype(np.float32)
    comb = torch.from_numpy(c).to(device=raw.device, dtype=raw.dtype)
    return pre, post, comb


def _native_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    """pre/post on the device, the hc x hc Sinkhorn in the hand-written kernel."""
    from . import native

    hc = int(hc_mult)
    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:].view(b, s, hc, hc) * hc_scale[2]
            + hc_base[2 * hc:].view(hc, hc)).contiguous()
    out = torch.empty_like(comb)
    n = int(b * s)
    fn = native.kernel("sinkhorn", "sinkhorn_hc")
    native.launch(fn, ((n + 63) // 64,), (64,),
                  [comb, out, n, hc, int(iters), float(eps)])
    return pre, post, out


def _sinkhorn_body(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    )
    comb = (mixes[..., 2 * hc_mult:].view(b, s, hc_mult, hc_mult) * hc_scale[2]
            + hc_base[2 * hc_mult:].view(hc_mult, hc_mult))
    comb = torch.softmax(comb, dim=-1) + eps
    # In-place: the tensor is [b, s, 2, 2] (four elements), so every out-of-place op here
    # allocates and copies far more than it computes -- 20 iterations x 4 ops x 2 calls x
    # 40 layers is ~10k ops and ~70ms of host time per token. div_ writes the same values
    # in the same order, so the result is bit-identical.
    comb.div_(comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb.div_(comb.sum(dim=-1, keepdim=True) + eps)
        comb.div_(comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


# The eager form issues 20 iterations x (2 sums + 2 divides) per call, twice per layer
# and 40 layers deep: ~6400 launches and ~70ms of host time for a 2x2 matrix. Inductor
# fuses it into a couple of kernels. Compilation is one-off per (hc_mult, iters) and is
# skipped entirely when torch.compile cannot build on this box.
_SINKHORN_COMPILED: dict[tuple[int, int], object] = {}
_SINKHORN_GRAPHS: dict[tuple[int, int], object] = {}


class _SinkhornGraph:
    """Replay the 2x2 Sinkhorn as one CUDA graph.

    The eager form issues ~160 kernels per call (20 iterations x 4 ops, twice per layer,
    40 layers) for a four-element matrix; a captured graph replays the whole thing with a
    single submit, and the only per-call work is copying the 24 coefficients in and the
    12 out.
    """

    def __init__(self, hc_mult: int, iters: int, eps: float):
        self.hc = int(hc_mult)
        self.iters = int(iters)
        self.eps = float(eps)
        total = hc_mult * 3 + hc_mult * hc_mult
        dev = "cuda"
        self.inp = torch.zeros(1, 1, total, dtype=torch.float32, device=dev)
        self.scale = torch.zeros(3, dtype=torch.float32, device=dev)
        self.base = torch.zeros(total, dtype=torch.float32, device=dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                outs = _sinkhorn_body(self.inp, self.scale, self.base, hc_mult, iters, eps)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outs = _sinkhorn_body(self.inp, self.scale, self.base, hc_mult, iters, eps)

    def __call__(self, mixes, scale, base):
        self.inp.copy_(mixes.detach().float().view(1, 1, -1))
        self.scale.copy_(scale.detach().float())
        self.base.copy_(base.detach().float())
        self.graph.replay()
        return self.outs


def _sinkhorn_graph(hc_mult: int, iters: int, eps: float):
    key = (int(hc_mult), int(iters))
    g = _SINKHORN_GRAPHS.get(key)
    if g is not None:
        return g
    g = False
    try:
        g = _SinkhornGraph(hc_mult, iters, eps)
    except Exception:
        g = False
    _SINKHORN_GRAPHS[key] = g
    return g


def _sinkhorn_fn(hc_mult: int, iters: int, eps: float):
    key = (int(hc_mult), int(iters))
    fn = _SINKHORN_COMPILED.get(key)
    if fn is not None:
        return fn
    fn = False
    try:
        import torch._dynamo  # noqa: F401  (present only with a full torch build)

        def body(mixes, hc_scale, hc_base):
            return _sinkhorn_body(mixes, hc_scale, hc_base, hc_mult, iters, eps)

        cand = torch.compile(body, dynamic=False, fullgraph=True)
        # compile now, on the real shape, so the first token is not the one paying for it
        probe = torch.zeros(1, 1, (hc_mult * 2) + hc_mult * hc_mult, device="cuda")
        cand(probe, torch.zeros(3, device="cuda"), torch.zeros((hc_mult * 2) + hc_mult * hc_mult, device="cuda"))
        torch.cuda.synchronize()
        fn = cand
    except Exception:
        fn = False
    _SINKHORN_COMPILED[key] = fn
    return fn


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch port of kernel.hc_split_sinkhorn (pre / post / doubly-stochastic comb)."""
    # Hand-written kernel first: the eager form is ~160 launches of a 4-element kernel per
    # call, twice per layer, 40 layers deep. Verified bit-identical to the eager result.
    if mixes.is_cuda and __import__("os").environ.get("TI_SINKHORN") != "eager":
        try:
            return _native_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
        except Exception as _exc:
            if __import__("os").environ.get("TI_SINKHORN_DEBUG"):
                print(f"[sinkhorn] native path failed: {_exc!r}", flush=True)
    return _sinkhorn_body(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


_NATIVE = {"ok": None}


def _native_ok() -> bool:
    """True when the hand-written hc kernels are built."""
    if _NATIVE["ok"] is None:
        try:
            from . import native
            native.kernel("hc", "hc_pre_f")
            _NATIVE["ok"] = True
        except Exception:
            _NATIVE["ok"] = False
    return _NATIVE["ok"]


def _kern(name: str):
    from . import native
    return native.kernel("hc", name)


def _launch(fn, grid, block, args):
    from . import native
    native.launch(fn, grid, block, args, 0)


class HCBlockMixin:
    """The coefficient machinery every Block uses (official Block.hc_mixes/hc_pre/hc_post)."""

    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    hc_attn_fn: torch.Tensor
    hc_attn_scale: torch.Tensor
    hc_attn_base: torch.Tensor
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """x: [b,s,hc,d] -> pre/post/comb coefficients (fused kernel on CUDA)."""
        global _FUSED_MIXES_OK
        if x.is_cuda and _FUSED_MIXES_OK:
            try:
                from .hc_fused import fused_hc_mixes, fused_sinkhorn
                b, s = x.shape[0], x.shape[1]
                x2 = x.reshape(b * s, -1).to(torch.bfloat16)
                raw = fused_hc_mixes(x2, hc_fn.float(), self.hc_eps)  # [b*s, 24] fp32
                pre, post, comb = fused_sinkhorn(raw, hc_scale.float(), hc_base.float(),
                                                 self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
                return (pre.view(b, s, self.hc_mult), post.view(b, s, self.hc_mult),
                        comb.view(b, s, self.hc_mult, self.hc_mult))
            except Exception:
                _FUSED_MIXES_OK = False
        # NOTE: a fused hc_mixes kernel (native/hc.cu, hc_mixes_f/_bf) exists and matches the
        # eager projection to ~2.9e-6, but that is not bit-identical: the torch path goes
        # through F.linear (cuBLAS tile order) while the kernel reduces with a warp tree.
        # Enabling it changed the greedy answer from "中国的首都是北京" to "中国的首都北京",
        # so it stays off -- tier 2 is defined as full precision and the answer must not
        # move. TI_HC_MIXES_KERNEL=1 re-enables it for benchmarking only.
        if (__import__("os").environ.get("TI_HC_MIXES_KERNEL") == "1"
                and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and _native_ok()):
            try:
                b, s = x.shape[0], x.shape[1]
                hc, d = x.shape[2], x.shape[3]
                k_dim = hc * d
                j_dim = int(hc_fn.shape[0])
                x2 = x.reshape(b * s, k_dim)
                fn32 = hc_fn.float().contiguous()
                out = torch.empty((b * s, j_dim), dtype=torch.float32, device=x.device)
                bfs = x.dtype == torch.bfloat16
                _launch(_kern("hc_mixes_bf" if bfs else "hc_mixes_f"),
                        (b * s,), (256,),
                        [x2.contiguous() if bfs else x2.contiguous().float(), fn32, out,
                         b * s, k_dim, j_dim, float(self.hc_eps), 0])
                return hc_split_sinkhorn(out.view(b, s, j_dim), hc_scale, hc_base,
                                         self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
            except Exception:
                pass
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        # One kernel instead of mul+sum+to. The eager form is 5 launches per call and this
        # runs twice per layer (80 calls per token, ~400 launches); the kernel does the
        # same fp32 accumulation and the same final cast, one thread per output element,
        # so it is bit-identical and needs no chain of temporaries.
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and _native_ok():
            try:
                b, s, hc, d = x.shape
                xc = x.reshape(b * s, hc, d)
                if xc.dtype == torch.bfloat16:
                    xc = xc.contiguous()
                    xf = None
                    out = torch.empty((b * s, d), dtype=torch.bfloat16, device=x.device)
                    fn = _kern("hc_pre_bf")
                    n = b * s * d
                    _launch(fn, ((n + 255) // 256,), (256,),
                            [xc, pre_mix.reshape(b * s, hc).float().contiguous(), out,
                             b * s, hc, d])
                else:
                    out = torch.empty((b * s, d), dtype=torch.float32, device=x.device)
                    fn = _kern("hc_pre_f")
                    n = b * s * d
                    _launch(fn, ((n + 255) // 256,), (256,),
                            [xc.contiguous(), pre_mix.reshape(b * s, hc).float().contiguous(),
                             out, b * s, hc, d])
                return out.view(b, s, d)
            except Exception:
                pass
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """post * x + comb @ residual, HC-space residual update.

        Measured on this box: the torch path is 0.06ms for a 164KB problem. The fused
        kernel raises "RES shape[0] expected 4, got 1" for single-token steps, so its
        availability is latched once per process rather than re-raised per call.
        """
        global _FUSED_POST_OK
        # Deterministic native fast path: post*x + comb@residual, one thread per output
        # element with the same fp32 accumulation order the eager expression uses. The
        # eager form is 6 launches (unsqueeze x2, mul x2, sum, type_as) twice per layer.
        if (x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
                and residual.shape == x.shape and _native_ok()):
            try:
                b, s, hc, d = x.shape
                xc = x.reshape(b * s, hc, d).contiguous()
                rc = residual.reshape(b * s, hc, d).contiguous()
                pc = post.reshape(b * s, hc, hc).float().contiguous()
                cc = comb.reshape(b * s, hc, hc).float().contiguous()
                bfs = x.dtype == torch.bfloat16
                out = torch.empty((b * s, hc, hc, d), dtype=x.dtype, device=x.device)
                n = b * s * hc * hc * d
                _launch(_kern("hc_post_bf" if bfs else "hc_post_f"),
                        ((n + 255) // 256,), (256,),
                        [xc, rc, pc, cc, out, b * s, hc, d])
                return out.view(b, s, hc, hc, d)
            except Exception:
                pass
        import os as _os
        if x.is_cuda and _FUSED_POST_OK and _os.environ.get("TI_HC_NOFUSE") != "1":
            try:
                from .hc_fused import fused_hc_post
                if _os.environ.get("TI_HC_T") == "1":
                    import time as _t
                    _t0 = _t.perf_counter()
                    y = fused_hc_post(x, residual, post, comb)
                    self._hc_f = getattr(self, "_hc_f", [0.0, 0])
                    self._hc_f[0] += (_t.perf_counter() - _t0) * 1e3
                    self._hc_f[1] += 1
                    if self._hc_f[1] == 8 and getattr(self, "layer_id", -1) == 5:
                        print(f"[hc_post fused] mean={self._hc_f[0]/self._hc_f[1]:.2f}ms", flush=True)
                    return y
                return fused_hc_post(x, residual, post, comb)
            except Exception as _e:
                _FUSED_POST_OK = False
                import os as _os
                if _os.environ.get("TI_HC_DEBUG"):
                    print(f"[hc_post] fused kernel disabled ({_e!r})", flush=True)
        import os as _os
        if _os.environ.get("TI_HC_T") == "1":
            import time as _t
            _t0 = _t.perf_counter()
            y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
            _dt = (_t.perf_counter() - _t0) * 1e3
            self._hc_t = getattr(self, "_hc_t", [0.0, 0])
            self._hc_t[0] += _dt
            self._hc_t[1] += 1
            if self._hc_t[1] == 8 and getattr(self, "layer_id", -1) in (5, 20):
                print(f"[hc_post L{self.layer_id}] contig x={x.is_contiguous()} res={residual.is_contiguous()} "
                      f"post={post.is_contiguous()} dtype x={x.dtype} post={post.dtype} "
                      f"x={tuple(x.shape)} res={tuple(residual.shape)} "
                      f"mean={self._hc_t[0]/self._hc_t[1]:.2f}ms", flush=True)
            return y.type_as(x)
        # fp32 coefficients: tier <= 2 keeps the full-precision arithmetic. The bf16
        # variant is a tier-3 trade (two fewer full-tensor conversions per call, 80 calls
        # per token).
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)
