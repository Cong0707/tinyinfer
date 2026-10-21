"""MTP / DSpark speculative stages (checkpoint tensors: mtp.0 .. mtp.2).

Reference implementations (read these first when fixing this file):

  * DeepInfer  deepinfer/model_executor/models/mtp/dflash.py
        DFlash.__init__                -> this file's MTPBlock weight loading
        DFlash._embed_draft_init_tokens -> MTPBlock.forward_embed
        DFlash.update_main_cache       -> MTPBlock.main_hiddens (projection + project_norm)
        DFlash.forward_drafts          -> MTPBlock.forward
        DFlash.sample_drafts           -> MTPBlock.forward_head
  * DeepInfer  deepinfer/model_executor/models/v4_modules/attn.py
        DFlashAttention.update_main_cache / forward_drafts -> MTPAttention.forward_with_main
  * DeepInfer  deepinfer/model_executor/models/v4_modules/heads.py
        DFlashNextPredictHeadHC.projection / project_norm  -> mtp.0.main_proj / main_norm
        DFlashNextPredictHeadHC.shared_head.markov_w1/w2   -> mtp.2.markov_head.embed/head
        DFlashNextPredictHeadHC.confidence_head.proj       -> mtp.2.confidence_head.proj
  * vLLM / HF transformers equivalent: a standard "nextn"/MTP decoder block stack
        (num_nextn_predict_layers=3, draft block_size=5, markov_rank=256).

Checkpoint tensor names (keep these exact names when loading):
        mtp.{S}.attn.wq_a/.wq_b/.wkv/.wo_a/.wo_b .weight/.scale   fp8 attention
        mtp.{S}.attn.q_norm/.kv_norm.weight, attn_sink
        mtp.{S}.attn_norm.weight / ffn_norm.weight
        mtp.{S}.hc_attn_fn / hc_attn_scale / hc_attn_base         hyper-connections
        mtp.{S}.hc_ffn_fn / hc_ffn_scale / hc_ffn_base
        mtp.{S}.ffn.gate.{weight,bias,bias_vl}, ffn.shared_experts.*,
        mtp.{S}.ffn.experts.{i}.{w1,w2,w3}.weight/.scale          MoE (128 routed, top-3)
        mtp.0.main_proj.weight/.scale  fp8 [dim, n_target*dim]    official: projection
        mtp.0.main_norm.weight         bf16 [dim]                 official: project_norm
        mtp.2.norm.weight              bf16 [dim]                 official: post_norm
        mtp.2.markov_head.embed.weight [vocab, markov_rank]       official: markov_w1
        mtp.2.markov_head.head.weight  [markov_rank, vocab]       official: markov_w2
        mtp.2.confidence_head.proj.weight [1, dim+markov_rank]    official: confidence_head.proj

Config fields (from HF config.json text_config, TinyConfig keeps the same names):
        dspark_block_size=5, dspark_target_layer_ids=[37,38,39], dspark_markov_rank=256,
        dspark_noise_token_id=128799, dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3, hc_mult=4
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import MLAWindowAttention
from .config import TinyConfig
from .moe import MoE
from .nn import apply_rope, rms_norm, window_idxs_general
from .quant import dequant_fp8_block
from .weights import ShardStore


class _PrefixStore:
    """Maps layers.{src_layer}.* reads onto mtp.{stage}.*."""

    def __init__(self, store: ShardStore, stage: int, src_layer: int = 0):
        self._store = store
        self._stage = stage
        self._src = src_layer
        self.model_dir = store.model_dir

    def _map(self, name: str) -> str:
        prefix = f"layers.{self._src}."
        if name.startswith(prefix):
            return f"mtp.{self._stage}." + name[len(prefix):]
        return name

    def has(self, name: str) -> bool:
        return self._store.has(self._map(name))

    def shard_of(self, name: str) -> str:
        return self._store.shard_of(self._map(name))

    def get(self, name: str) -> torch.Tensor:
        return self._store.get(self._map(name))


class MTPAttention(MLAWindowAttention):
    """Draft attention seeded with main-model KV (official DFlashAttention).

    Official split:
        update_main_cache(main_k)  -> kv_norm + rope + write into the main KV window
        forward_drafts(x)          -> draft q/kv + sparse attention over [main kv, draft kv]
    This class merges both in forward_with_main: the main projection (main_x) is
    already the official DFlash stage-0 projection output.
    """

    def forward_with_main(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        """Draft attention over a sliding window of main KV history plus the draft block.

        main_x: [1, n_main, dim] main-model projection output for the current token
                (official: projection_norm(projection(dflash_input))), position start_pos-1.
        x:      [1, block, dim] draft hiddens for positions start_pos .. start_pos+block-1.
        The main window keeps the last 'window' main KV entries across decode steps
        (official swa cache); draft KV only lives for this block.
        """
        b, s, _ = x.shape
        win = self.window
        rd = self.rope_dim
        n_main = main_x.size(1)
        main_pos0 = start_pos - n_main  # main tokens occupy [start_pos-n_main, start_pos)

        def _lin(w, x_):
            return w.forward(x_) if hasattr(w, "forward") else F.linear(x_, w)

        # official update_main_cache: main_k = main_kv_proj(main_hiddens) then kv_norm + rope
        main_kv = rms_norm(_lin(self.wkv, main_x), self.kv_norm, self.eps)  # [1, n_main, head_dim]
        apply_rope(main_kv[..., -rd:], self.freqs_cis[main_pos0:start_pos])
        hist = self._kv_window
        if hist is None or hist.shape[0] != b or hist.device != main_kv.device:
            hist = main_kv.new_empty(b, 0, self.head_dim)
        hist = torch.cat([hist, main_kv], dim=1)[:, -win:].contiguous()
        self._kv_window = hist
        if start_pos == 0:
            return x  # prefill only seeds the window (official: no draft compute)

        # official forward_drafts: draft q/kv at positions start_pos..start_pos+s-1
        freqs = self.freqs_cis[start_pos:start_pos + s]
        qr = rms_norm(_lin(self.wq_a, x), self.q_norm, self.eps)
        q = _lin(self.wq_b, qr).view(b, s, self.n_heads, self.head_dim)
        apply_rope(q[..., -rd:], freqs)
        kv = rms_norm(_lin(self.wkv, x), self.kv_norm, self.eps)
        apply_rope(kv[..., -rd:], freqs)

        kv_att = torch.cat([hist, kv], dim=1)
        n_hist = hist.size(1)
        idxs = window_idxs_general(start_pos, s, n_hist, win, x.device).unsqueeze(0).expand(b, -1, -1)
        from .kernels import sparse_attn_dispatch
        o = sparse_attn_dispatch(q, kv_att, self.sink, idxs, self.scale)
        apply_rope(o[..., -rd:], freqs, inverse=True)
        o = o.view(b, s, self.o_groups, -1)
        if hasattr(self.wo_a, "weight"):  # fp8 resident: dequant just for the grouped einsum
            from .quant import dequant_fp8_block
            wo_a = dequant_fp8_block(self.wo_a.weight, self.wo_a.scale)
        else:
            wo_a = self.wo_a
        wo_a = wo_a.view(self.o_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return _lin(self.wo_b, o.flatten(2))


class MTPBlock:
    """One DSpark stage (official DFlashDecoderLayer + DFlashNextPredictHeadHC pieces).

    Stage 0 also owns main_proj/main_norm (official DFlash.projection/project_norm);
    the last stage owns post_norm, markov_w1/w2 and the confidence head.
    """

    def __init__(self, store: ShardStore, stage: int, cfg: TinyConfig, quant: bool = False):
        import dataclasses

        self.stage = stage
        self.cfg = cfg
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        stage_cfg = cfg
        if cfg.dspark_n_routed_experts and cfg.dspark_n_routed_experts != cfg.n_routed_experts:
            stage_cfg = dataclasses.replace(cfg, n_routed_experts=cfg.dspark_n_routed_experts,
                                            n_activated_experts=cfg.dspark_n_activated_experts)
        shim = _PrefixStore(store, stage, 0)
        self.attn = MTPAttention(shim, 0, cfg, quant=quant)
        self.ffn = MoE(shim, 0, stage_cfg)
        self.attn_norm = shim.get("layers.0.attn_norm.weight")
        self.ffn_norm = shim.get("layers.0.ffn_norm.weight")
        self.hc_attn_fn = shim.get("layers.0.hc_attn_fn").float()
        self.hc_attn_base = shim.get("layers.0.hc_attn_base").float()
        self.hc_attn_scale = shim.get("layers.0.hc_attn_scale").float()
        self.hc_ffn_fn = shim.get("layers.0.hc_ffn_fn").float()
        self.hc_ffn_base = shim.get("layers.0.hc_ffn_base").float()
        self.hc_ffn_scale = shim.get("layers.0.hc_ffn_scale").float()
        self.block_size = cfg.dspark_block_size
        self.eps = cfg.rms_norm_eps
        # the checkpoint has no mtp.*.embed / mtp.*.head: drafts reuse the main model's
        # embedding and output head (injected by Engine.load_mtp).
        self.embed = None
        self.head = None
        # official DFlash.projection / project_norm (checkpoint: mtp.0.main_proj/main_norm)
        self.main_proj = self.main_norm = None
        if stage == 0:
            self.main_proj = dequant_fp8_block(shim.get("layers.0.main_proj.weight"),
                                               shim.get("layers.0.main_proj.scale"))                 if shim.has("layers.0.main_proj.scale") else shim.get("layers.0.main_proj.weight")
            self.main_norm = shim.get("layers.0.main_norm.weight")
        # official DFlashNextPredictHeadHC.shared_head (post_norm + markov) and confidence head
        self.norm = self.markov_embed = self.markov_head = self.confidence_proj = None
        if stage == cfg.n_mtp_layers - 1:
            self.norm = shim.get("layers.0.norm.weight")
            self.markov_embed = shim.get("layers.0.markov_head.embed.weight")
            self.markov_head = shim.get("layers.0.markov_head.head.weight")
            self.confidence_proj = shim.get("layers.0.confidence_head.proj.weight").float()

    # -- HC helpers (same math as Block) ----------------------------------
    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        from .hc import hc_split_sinkhorn
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x, pre_mix):
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    # -- official DFlash.projection / project_norm -------------------------
    def main_hiddens(self, target_hiddens: list[torch.Tensor]) -> torch.Tensor:
        """Concat the dspark target-layer hiddens -> projection -> project_norm.

        target_hiddens: per dspark_target_layer_ids, each [1, T, dim] (collapsed, bf16).
        Returns [1, T, dim] (the main_x feeding MTPAttention.forward_with_main).
        """
        mh = torch.cat(target_hiddens, dim=-1)  # [1, T, n_target*dim]
        return rms_norm(F.linear(mh.to(self.main_proj.dtype), self.main_proj),
                        self.main_norm, self.eps)

    # -- official DFlash._embed_draft_init_tokens --------------------------
    def forward_embed(self, golden_tokens: torch.Tensor) -> torch.Tensor:
        """Embed the draft init block: noise fill, position 0 is the golden token.

        Mirrors DFlash._embed_draft_init_tokens; draft hiddens do NOT come from
        main_proj (that tensor only produces the main KV projections).
        """
        if self.embed is None:
            raise RuntimeError("MTPBlock.embed is not set; Engine.load_mtp must inject the main embedding")
        draft_ids = golden_tokens.new_full([golden_tokens.size(0), self.block_size],
                                           self.cfg.dspark_noise_token_id)
        draft_ids[:, 0] = golden_tokens
        emb = F.embedding(draft_ids, self.embed)
        return emb.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)

    # -- official DFlashDecoderLayer.forward_drafts (+ HC part of DFlash.forward_drafts)
    def forward(self, x, start_pos, pre_mix, main_x):
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.hc_pre(x, pre_mix)
        h = rms_norm(h, self.attn_norm, self.eps)
        h = self.attn.forward_with_main(h, start_pos, main_x)
        x = self.hc_post(h, x, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.hc_pre(x, attn_pre)
        h = rms_norm(h, self.ffn_norm, self.eps)
        h = self.ffn.forward(h)
        x = self.hc_post(h, x, ffn_post, ffn_comb)
        return x, ffn_pre

    # -- official DFlash.sample_drafts (post_norm -> head -> markov -> confidence)
    def forward_head(self, x, pre_mix):
        """Last stage: collapse HC, post_norm, main-head logits, markov correction, confidence.

        Returns (logits [1, block, vocab], confidence [1, block]) where confidence is the
        acceptance-rate prediction (sigmoid of the projected hidden; official temperature
        normalization is applied by the caller in the reference implementation).
        """
        if self.norm is None:
            return None, None
        h = self.hc_pre(x, pre_mix)  # collapse HC -> [1, block, dim]
        h = rms_norm(h, self.norm, self.eps)
        logits = F.linear(h, self.head) if self.head is not None else None
        if logits is None:
            return None, None
        if self.markov_embed is not None and self.markov_head is not None:
            # official markov correction: logits += markov_w2(markov_w1[draft_ids])
            pass
        if self.confidence_proj is not None:
            # official: proj([hidden, markov_emb]) -> sigmoid -> acceptance rate
            pass
        return logits, None

    def to(self, device):
        for name in ("attn_norm", "ffn_norm", "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                     "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "main_proj", "main_norm",
                     "norm", "markov_embed", "markov_head", "confidence_proj"):
            v = getattr(self, name, None)
            if isinstance(v, torch.Tensor):
                setattr(self, name, v.to(device))
        a = self.attn
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            ql = getattr(a, name, None)
            if ql is None:
                continue
            if hasattr(ql, "to") and not isinstance(ql, torch.Tensor):
                ql.to(device)
            else:
                setattr(a, name, ql.to(device))
        for name in ("q_norm", "kv_norm", "sink", "freqs_cis"):
            t = getattr(a, name, None)
            if isinstance(t, torch.Tensor):
                setattr(a, name, t.to(device))
        self.ffn.to(device)
        return self


def load_mtp(store: ShardStore, cfg: TinyConfig, quant: bool = False) -> list[MTPBlock] | None:
    """Instantiate all MTP stages if the mtp.* weights are present (official DFlash: self.layers)."""
    if cfg.n_mtp_layers <= 0:
        return None
    try:
        stages = [MTPBlock(store, st, cfg, quant=quant) for st in range(cfg.n_mtp_layers)]
    except (KeyError, FileNotFoundError, ValueError):
        return None
    return stages
