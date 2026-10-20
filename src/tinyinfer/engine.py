"""tinyinfer engine: full-model driver on top of adapters + tiers + ExpertStore.

Owns model lifecycle: embedding, block residency (lazy per-layer load), generation.
The fetch/dequant/eviction policy lives in loading.py; this is the orchestration layer.
"""
from __future__ import annotations

from .adapters import ModelAdapter, open_adapter
from .config import TinyConfig
from .hc import hc_split_sinkhorn  # noqa: F401  (re-export for callers)
from .nn import rms_norm, softmax_sample
from .model import Block
from .tiers import OptLevel, TierConfig, plan

import torch
import torch.nn.functional as F


class Engine:
    def __init__(
        self,
        adapter: ModelAdapter | str,
        *,
        tier: OptLevel | TierConfig = OptLevel.BALANCED,
        expert_store=None,
        max_layers: int | None = None,
        allow_missing_layers: bool = False,
        device: str = "cpu",
    ):
        self.device = device
        self.adapter = open_adapter(adapter) if isinstance(adapter, str) else adapter
        self.cfg = self.adapter.load_config()
        self.store = self.adapter.store
        self.tier = plan(tier) if isinstance(tier, OptLevel) else tier
        self.expert_store = expert_store
        self.max_layers = self.cfg.n_layers if max_layers is None else min(max_layers, self.cfg.n_layers)
        self.allow_missing_layers = allow_missing_layers
        self._blocks: dict[int, Block] = {}
        self.embed = self.store.get("embed.weight").to(device)
        self.out_norm = self.store.get("norm.weight").to(device)
        self.head = self.store.get("head.weight").to(device)
        self.eps = self.cfg.rms_norm_eps

    # -- residency ---------------------------------------------------------
    def available_layers(self) -> int:
        """How many consecutive layers are fully readable on disk right now."""
        from safetensors import safe_open

        n = 0
        for i in range(self.cfg.n_layers):
            name = f"layers.{i}.attn.wq_a.weight"
            if not self.store.has(name):
                break
            try:
                with safe_open(str(self.store.model_dir / self.store.shard_of(name)), framework="pt"):
                    pass
            except Exception:
                break
            n = i + 1
        return n

    def block(self, layer_id: int) -> Block:
        b = self._blocks.get(layer_id)
        if b is None:
            b = Block(self.store, layer_id, self.cfg, expert_store=self.expert_store).to(self.device)
            self._blocks[layer_id] = b
        return b

    def reset(self) -> None:
        for b in self._blocks.values():
            if hasattr(b.attn, "reset"):
                b.attn.reset()
            b.ffn._expert_cache.clear()

    # -- forward -----------------------------------------------------------
    def forward(self, ids: torch.Tensor, start_pos: int = 0):
        n = self.max_layers
        if self.allow_missing_layers:
            n = min(n, self.available_layers())
        h = self.embed[ids]
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32, device=h.device)
        pre_mix[:, :, 0] = 1.0
        for i in range(n):
            h, pre_mix = self.block(i).forward(h, start_pos, pre_mix)
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.eps)
        return torch.nn.functional.linear(h, self.head).float()

    def generate(self, prompt_ids, max_new_tokens: int = 16, temperature: float = 0.0, eos_token_id=None):
        """Greedy/sampled generation loop (prefill once, then decode step by step)."""
        ids = list(prompt_ids)
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            out = []
            for step in range(max_new_tokens):
                nid = int(softmax_sample(logits[0, -1], temperature))
                out.append(nid)
                if eos_token_id is not None and nid == eos_token_id:
                    break
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=len(ids) + step)
        return out

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict:
        out = {
            "layers_resident": len(self._blocks),
            "max_layers": self.max_layers,
            "n_layers": self.cfg.n_layers,
            "tier": self.tier.level.name,
        }
        if self.expert_store is not None:
            out["experts"] = self.expert_store.stats()
        return out
