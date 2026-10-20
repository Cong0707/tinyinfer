"""TinyConfig: HF config.json -> tinyinfer runtime config (single-host, no TP)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TinyConfig:
    # text backbone
    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    qk_rope_head_dim: int
    rope_theta: float
    rope_scaling: dict | None
    max_position_embeddings: int
    compress_ratios: tuple[int, ...]
    compress_rope_theta: float
    sliding_window: int
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    # moe
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    scoring_func: str
    norm_topk_prob: bool
    route_scale: float
    swiglu_limit: float
    # hyper-connections
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    # engram
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_id: int
    engram_compressed_vocab_size: int
    # dspark / mtp
    n_mtp_layers: int
    dspark_block_size: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_n_activated_experts: int
    # vision
    vision_enabled: bool
    vision_cfg: dict = field(default_factory=dict)
    image_token_id: int = 129264
    # quantization
    dtype: str = "bf16"
    expert_dtype: str = "fp4"
    weight_block_size: tuple[int, int] = (32, 32)
    scale_fmt: str = "ue8m0"
    rms_norm_eps: float = 1e-20
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "TinyConfig":
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        tc = cfg.get("text_config", cfg)
        qc = cfg.get("quantization_config", {})
        vc = cfg.get("vision_config", {})
        rs = tc.get("rope_scaling") or None
        return cls(
            vocab_size=int(tc["vocab_size"]),
            dim=int(tc["hidden_size"]),
            moe_inter_dim=int(tc["moe_intermediate_size"]),
            n_layers=int(tc["num_hidden_layers"]),
            n_heads=int(tc["num_attention_heads"]),
            head_dim=int(tc["head_dim"]),
            q_lora_rank=int(tc["q_lora_rank"]),
            o_lora_rank=int(tc["o_lora_rank"]),
            o_groups=int(tc["o_groups"]),
            qk_rope_head_dim=int(tc["qk_rope_head_dim"]),
            rope_theta=float(tc["rope_theta"]),
            rope_scaling=rs,
            max_position_embeddings=int(tc.get("max_position_embeddings", 1048576)),
            compress_ratios=tuple(tc.get("compress_ratios", ())),
            compress_rope_theta=float(tc.get("compress_rope_theta", tc["rope_theta"])),
            sliding_window=int(tc.get("sliding_window", 0)),
            kv_source_layer_ids=tuple(tc.get("kv_source_layer_ids", ())),
            index_source_layer_ids=tuple(tc.get("index_source_layer_ids", ())),
            index_n_heads=int(tc.get("index_n_heads", 0)),
            index_head_dim=int(tc.get("index_head_dim", 0)),
            index_topk=int(tc.get("index_topk", 0)),
            candidate_source_layer_id=int(tc.get("candidate_source_layer_id", -1)),
            candidate_topk_blocks=int(tc.get("candidate_topk_blocks", 0)),
            candidate_block_size=int(tc.get("candidate_block_size", 0)),
            n_routed_experts=int(tc["n_routed_experts"]),
            n_shared_experts=int(tc.get("n_shared_experts", 1)),
            n_activated_experts=int(tc["num_experts_per_tok"]),
            scoring_func=str(tc.get("scoring_func", "sqrtsoftplus")),
            norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
            route_scale=float(tc.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(tc.get("swiglu_limit", 0.0)),
            hc_mult=int(tc.get("hc_mult", 1)),
            hc_sinkhorn_iters=int(tc.get("hc_sinkhorn_iters", 20)),
            hc_eps=float(tc.get("hc_eps", 1e-6)),
            engram_layer_ids=tuple(tc.get("engram_layer_ids", ())),
            engram_num_embeddings=tuple(tc.get("engram_num_embeddings", ())),
            engram_max_ngram_size=int(tc.get("engram_max_ngram_size", 0)),
            engram_vocab_size=int(tc.get("engram_vocab_size", 0)),
            engram_n_heads=int(tc.get("engram_n_heads", 0)),
            engram_head_dim=int(tc.get("engram_head_dim", 0)),
            engram_pad_id=int(tc.get("engram_pad_token_id", 2)),
            engram_compressed_vocab_size=int(tc.get("engram_compressed_vocab_size", 0)),
            n_mtp_layers=int(tc.get("num_nextn_predict_layers", 0)),
            dspark_block_size=int(tc.get("dspark_block_size", 0)),
            dspark_target_layer_ids=tuple(tc.get("dspark_target_layer_ids", ())),
            dspark_markov_rank=int(tc.get("dspark_markov_rank", 256)),
            dspark_n_routed_experts=int(tc.get("dspark_n_routed_experts", 0)),
            dspark_n_activated_experts=int(tc.get("dspark_num_experts_per_tok", 0)),
            vision_enabled=bool(vc.get("num_hidden_layers", 0)),
            vision_cfg=vc,
            image_token_id=int(cfg.get("image_token_id", 129264)),
            dtype=str(cfg.get("dtype", "bfloat16")),
            expert_dtype=str(qc.get("expert_dtype", "fp4")),
            weight_block_size=tuple(qc.get("weight_block_size", (32, 32))),
            scale_fmt=str(qc.get("scale_fmt", "ue8m0")),
            rms_norm_eps=float(tc.get("rms_norm_eps", 1e-20)),
            tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
        )

    def arch_family(self) -> str:
        """Architecture family used by tinyinfer.adapters.open_adapter()."""
        return "deepseek_v41"

    def summary(self) -> str:
        return (
            f"layers={self.n_layers} dim={self.dim} heads={self.n_heads} head_dim={self.head_dim} "
            f"experts={self.n_routed_experts}/{self.n_activated_experts} "
            f"engram_layers={self.engram_layer_ids} mtp={self.n_mtp_layers} "
            f"vision={self.vision_enabled}"
        )
