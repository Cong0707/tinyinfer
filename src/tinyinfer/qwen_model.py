"""Qwen3.5 as an engine model.

The engine contributes what is expensive and architecture-blind: streaming weights from
disk, layer residency, prefetch, the CUDA graph, KV ownership. The layer arithmetic is
NOT reimplemented here -- it is delegated to the reference implementation of the same
architecture, which is the only way to stay correct as checkpoints change. What this
adapter adds is the mapping from that implementation onto the engine's per-layer
lifecycle: a layer object the engine can move between CPU and GPU, whose weights come
from the checkpoint on demand.

That split is what lets a 51.8GB checkpoint run on an 8GB card: nothing here needs the
whole model resident, because the engine decides what is resident and this hands it one
layer at a time.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import torch


class QwenLayer:
    """One Qwen3.5 decoder layer whose weights live wherever the engine put them.

    Wraps the reference layer module. `to()` moves that module, so the engine's
    residency plan works unchanged: a resident layer stays on the card, a streamed one
    is moved in for its turn and out again.
    """

    def __init__(self, module, store, layer_id: int, prefix: str):
        self.module = module
        self.store = store
        self.layer_id = layer_id
        self.prefix = prefix
        self.device = "cpu"

    def _weight_paths(self) -> dict[str, Any]:
        """name -> shard; what the streaming planner measures and the ring prefetches."""
        return {n: self.store.weight_map[n]
                for n in self.store.weight_map if n.startswith(self.prefix)}

    def materialize(self) -> None:
        """Read this layer's tensors out of the checkpoint into the meta module.

        Done once per layer, on first use. After this the layer is a normal module: the
        engine can move it to the card and run it, and move it out again when the next
        layer needs the room.
        """
        if getattr(self, "_loaded", False):
            return
        import torch.nn as nn

        flat = {n[len(self.prefix):]: n for n in self.store.weight_map
                if n.startswith(self.prefix)}
        own = dict(self.module.named_parameters())
        own.update(self.module.named_buffers())
        missing = []
        for short, full in flat.items():
            target = own.get(short)
            tensor = self.store.get(full)
            if target is None:
                missing.append(short)
                continue
            # the reference loader casts to the model dtype; the checkpoint stores a few
            # tensors wider than the module expects (the gated-norm weight is fp32 on
            # disk), and leaving it wide changed the results
            if target.is_floating_point() and tensor.dtype != target.dtype:
                tensor = tensor.to(target.dtype)
            # the module was built on the meta device, so there is no storage to copy
            # into: swap the parameter for a real tensor instead
            parent_name, _, leaf = short.rpartition(".")
            parent = self.module.get_submodule(parent_name) if parent_name else self.module
            if isinstance(target, nn.Parameter):
                setattr(parent, leaf, nn.Parameter(tensor, requires_grad=False))
            else:
                setattr(parent, leaf, tensor)
        self._loaded = True
        self.missing = missing

    def to(self, device: str):
        if not getattr(self, "_loaded", False):
            self.materialize()
        if device != self.device:
            self.module.to(device)
            self.device = device
        return self

    def forward(self, hidden, cache=None, position=None):
        if not getattr(self, "_loaded", False):
            self.materialize()
        out = self.module(hidden_states=hidden, past_key_values=cache,
                          attention_mask=None, position_ids=position, use_cache=True)
        return out[0] if isinstance(out, tuple) else out


class QwenModel:
    """A Qwen3.5 checkpoint behind the engine's model contract."""

    family = "qwen3_5"
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    head_key = None

    def __init__(self, store, cfg: dict):
        self.store = store
        self.cfg = cfg
        self.kinds = list(cfg["layer_types"])
        self._layers: dict[int, QwenLayer] = {}
        self._built = None
        self._lock = threading.Lock()

    @classmethod
    def open(cls, model_dir) -> "QwenModel":
        from .config import TinyConfig  # noqa: F401
        from .weights import ShardStore

        store = ShardStore(model_dir)
        raw = json.loads((store.model_dir / "config.json").read_text(encoding="utf-8"))
        raw = raw.get("text_config", raw)
        return cls(store, raw)

    def load_config(self):
        from .config import TinyConfig
        return TinyConfig.from_hf(self.store.model_dir)

    @property
    def n_layers(self) -> int:
        return int(self.cfg["num_hidden_layers"])

    def _skeleton(self):
        """The reference model on the meta device: shapes without allocating weights.

        This is what makes streaming possible for a 51.8GB checkpoint: the module tree is
        built for its structure only, and the engine moves individual layers onto the card
        as it needs them.
        """
        if self._built is not None:
            return self._built
        with self._lock:
            if self._built is not None:
                return self._built
            from transformers import Qwen3_5ForConditionalGeneration

            # build on the meta device: the module tree gets real shapes and zero bytes of
            # storage, which is what lets a 51.8GB checkpoint exist as a skeleton
            cfg = self._load_hf_config()
            # build on meta with the checkpoint's dtype: a meta module defaults to fp32,
            # and silently running the whole model wider than the reference changed the
            # results (and doubled the memory once materialised)
            dtype = getattr(cfg, "torch_dtype", None) or torch.bfloat16
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype, torch.bfloat16)
            with torch.device("meta"):
                self._built = Qwen3_5ForConditionalGeneration._from_config(cfg)
                self._built = self._built.to(dtype)
            self._built.eval()
        return self._built

    def _load_hf_config(self):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(str(self.store.model_dir),
                                          trust_remote_code=True)

    def build_layer(self, layer_id: int) -> QwenLayer:
        layer = self._layers.get(layer_id)
        if layer is None:
            model = self._skeleton()
            module = model.model.language_model.layers[layer_id]
            prefix = f"model.language_model.layers.{layer_id}."
            layer = self._layers[layer_id] = QwenLayer(module, self.store, layer_id, prefix)
        return layer

    def make_kv(self, capacity: int):
        from transformers.cache_utils import DynamicCache

        return DynamicCache()

    def forward(self, ids, start_pos: int, engine) -> torch.Tensor:
        raise NotImplementedError(
            "Qwen runs through the engine's generation loop, which drives one layer at a "
            "time; a whole-model forward is not part of the contract here.")