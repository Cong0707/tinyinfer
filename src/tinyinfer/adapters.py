"""Model adapters: tinyinfer should not be DeepSeek-only.

A ModelAdapter hides checkpoint layout and architecture differences behind a small
interface (config -> weight lookup -> block construction). The DeepSeek V4.1 adapter
is the first implementation; GGUF-based foreign models (ds4/flash-moe style) plug in
as a second adapter later.
"""
from __future__ import annotations

import os
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


def resolve_hf_snapshot(source: str | Path) -> Path:
    """Resolve a HF repo id / hub cache dir to a usable snapshot directory.

    Accepts (a) a flat downloaded checkpoint dir, (b) a HF hub snapshot dir, or
    (c) a repo id such as "deepseek-ai/DeepSeek-V4.1-Flash-1017" -- for (c) the
    already-downloaded snapshot is located with huggingface_hub (local only, no
    network access unless the caller asks for it).
    """
    p = Path(source)
    if (p / "model.safetensors.index.json").exists():
        return p
    if not p.exists():
        from huggingface_hub import snapshot_download
        try:
            snap = snapshot_download(str(source), local_files_only=True)
            return Path(snap)
        except Exception as exc:
            # hub cache may be present without the refs bookkeeping that
            # snapshot_download insists on; fall back to the raw snapshots dir.
            hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
            repo_dir = hub / f"models--{str(source).replace('/', '--')}"
            snaps = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").exists() else []
            for cand in snaps:
                if (cand / "model.safetensors.index.json").exists():
                    return cand
            raise FileNotFoundError(
                f"could not resolve HF repo {source!r}: not found in local cache (HF_HOME={hub})"
            ) from exc
    raise FileNotFoundError(f"unrecognized checkpoint layout under {p}")


def open_adapter(model_dir: str | Path) -> ModelAdapter:
    """Detect the checkpoint layout and return a matching adapter."""
    p = resolve_hf_snapshot(model_dir)
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
