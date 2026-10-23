"""What the engine needs from a model, and what a model must provide.

tinyinfer is not a DeepSeek engine. The engine owns the things that depend on the machine
(streaming weights from disk, prefetch rings, layer residency, slot KV ownership, CUDA
graph capture); a model supplies the things that depend on the architecture (weight
layout, one layer's forward, the embedding and head).

This module is the seam. Before it existed, engine.py imported model.Block directly and
Qwen had to bypass the engine entirely with its own in-process backend -- so the one
capability the engine exists for (running a checkpoint larger than VRAM) was unavailable
to every model except the one it was written for.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LayerLike(Protocol):
    """One transformer layer, as far as the engine is concerned.

    The engine moves these between CPU and GPU and asks for their weight paths so it can
    stream them; everything else about them is the model's business.
    """

    def _weight_paths(self) -> dict[str, Any]:
        """name -> path of the shard holding that weight (drive `streaming`)."""

    def to(self, device: str) -> "LayerLike":
        """Move resident weights; returns self for chaining."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """One layer's computation. The signature is the model's own."""


@runtime_checkable
class ModelLike(Protocol):
    """Everything the engine needs to run one checkpoint.

    Implementations live in the adapters, one per architecture family. The engine only
    ever talks to this interface.
    """

    def load_config(self) -> Any:
        """The runtime config (tinyinfer.config.TinyConfig or a family-specific one)."""

    def build_layer(self, layer_id: int) -> LayerLike:
        """Construct one layer. Called lazily; the engine decides residency."""

    def forward(self, ids: Any, start_pos: int, engine: Any) -> Any:
        """Run the model over ids[start_pos:], using engine for layer access and KV.

        `engine` exposes block(layer_id), the KV ring, and the sampling helpers, so a
        model can stream layers without knowing how streaming works.
        """

    def make_kv(self, capacity: int) -> Any:
        """Create this model's KV/state container (see tinyinfer.kv).

        Architectures differ here in a way the engine cannot guess: DeepSeek keeps a
        compressed latent per layer, Qwen alternates a KV window with a delta-rule
        recurrent state.
        """

    @property
    def n_layers(self) -> int: ...

    # -- checkpoint naming ------------------------------------------------
    # The engine loads the embedding, the final norm and the output head itself (they are
    # the largest single tensors and the residency plan singles them out), so it needs to
    # know what this checkpoint calls them. DeepSeek uses embed/norm/head; Qwen nests
    # them under model.language_model and ties the head to the embedding.
    embed_key: str
    norm_key: str
    head_key: str | None           # None when the head is the embedding


def check_model(model: Any) -> None:
    """Fail loudly and early if an adapter is missing part of the contract."""
    missing = [name for name in ("load_config", "build_layer", "forward")
               if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(
            f"{type(model).__name__} does not implement the model contract; missing "
            f"{missing}. See tinyinfer/protocol.py.")