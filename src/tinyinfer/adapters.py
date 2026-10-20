"""Model adapters: tinyinfer should not be DeepSeek-only.

A ModelAdapter hides checkpoint layout and architecture differences behind a small
interface (config -> weight lookup -> block construction). The DeepSeek V4.1 adapter
is the first implementation; GGUF-based foreign models (ds4/flash-moe style) plug in
as a second adapter later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from .config import TinyConfig
from .weights import ShardStore


class ModelAdapter(Protocol):
    name: str

    def load_config(self) -> TinyConfig: ...
    def has_tensor(self, name: str) -> bool: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...
    def build_block(self, layer_id: int, cfg: TinyConfig): ...


class HFDeepSeekV41Adapter:
    """HF safetensors layout for deepseek_v41 (our primary target)."""

    name = "deepseek_v41_hf"

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.store = ShardStore(self.model_dir)
        self.cfg = TinyConfig.from_hf(self.model_dir)

    def load_config(self) -> TinyConfig:
        return self.cfg

    def has_tensor(self, name: str) -> bool:
        return self.store.has(name)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.store.get(name)

    def build_block(self, layer_id: int, cfg: TinyConfig | None = None):
        from .model import Block
        return Block(self.store, layer_id, cfg or self.cfg)


def open_adapter(model_dir: str | Path) -> ModelAdapter:
    """Detect the checkpoint layout and return a matching adapter."""
    p = Path(model_dir)
    if (p / "model.safetensors.index.json").exists():
        cfg = TinyConfig.from_hf(p)
        # deepseek_v41 is the only architecture wired today; other HF MoE layouts will
        # register their own adapters here (qwen/glm/...), GGUF via a dedicated adapter.
        if cfg.arch_family() == "deepseek_v41":
            return HFDeepSeekV41Adapter(p)
        raise NotImplementedError(f"HF adapter for architecture family {cfg.arch_family()!r} not implemented yet")
    if any(p.glob("*.gguf")):
        raise NotImplementedError("GGUF adapter is not implemented yet (planned: ds4/flash-moe layout)")
    raise FileNotFoundError(f"unrecognized checkpoint layout under {p}")
