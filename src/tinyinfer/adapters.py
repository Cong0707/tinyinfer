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

    def __init__(self, model_dir):
        # model_dir may be a list / ';'-separated set of directories holding shards of
        # the SAME checkpoint spread over several disks (ShardStore validates it)
        self.store = ShardStore(model_dir)
        self.model_dir = self.store.model_dir
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


def _split_sources(model_dir) -> list[Path]:
    # model_dir may be one path, a list, or a ';'/',' separated set of directories. They
    # together hold ONE checkpoint striped across disks; ShardStore enforces that.
    raw = ([model_dir] if isinstance(model_dir, (str, Path))
           else [p for chunk in model_dir for p in str(chunk).split(";")])
    out: list[Path] = []
    for entry in raw:
        for part in str(entry).split(","):
            part = part.strip().strip('"')
            if part and Path(part).exists() and Path(part) not in out:
                out.append(Path(part))
    return out


def open_adapter(model_dir) -> ModelAdapter:
    sources = _split_sources(model_dir)
    if not sources:
        raise FileNotFoundError(f"no model directory exists among {model_dir!r}")
    p = resolve_hf_snapshot(sources[0])
    if len(sources) > 1 and (p / "model.safetensors.index.json").exists():
        cfg = TinyConfig.from_hf(p)
        if cfg.arch_family() == "deepseek_v41":
            return HFDeepSeekV41Adapter(sources)
        raise NotImplementedError(
            f"HF adapter for architecture family {cfg.arch_family()!r} not implemented yet")
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