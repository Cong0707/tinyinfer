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


class _LayerKv:
    """One layer's KV view on whichever device that layer runs on."""

    __slots__ = ("base", "layer_id", "device")

    def __init__(self, base, layer_id: int, device: str):
        self.base, self.layer_id, self.device = base, layer_id, device

    def layer(self, layer_id: int):
        return self.base.layer(layer_id)

    def get_seq_length(self):
        return self.base.get_seq_length()

    def update(self, keys, values, layer_id, kwargs=None):
        # the layer works on its own device; the cache stores where it is told to
        return self.base.update(keys, values, layer_id, kwargs)


class _LayerView:
    """One layer's slice of the cache, moved to the device the layer runs on."""

    def __init__(self, base, device: str, store: dict) -> None:
        self.base = base
        self.device = device
        self.store = store

    def layer(self, layer_id: int):
        return self.base.layer(layer_id)

    def update(self, keys, values, layer_id, kwargs=None):
        return self.base.update(keys, values, layer_id, kwargs)


def _layer_belongs(name: str) -> bool:
    """True for a decoder-layer tensor, whatever the checkpoint's prefix is."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return True
    return False


class QwenLayer:
    """One Qwen3.5 decoder layer behind the engine's streaming contract.

    The engine streams by asking a layer for (owner, attr, tensor) triples, packing them
    into one pinned buffer, and pointing the attributes at views of a device buffer. To be
    streamable at all, a Qwen layer has to answer in exactly that form -- the earlier
    version returned {name: shard}, which is why flat streaming reported "too many values
    to unpack" and every layer silently stayed on the host.
    """

    def __init__(self, module, store, layer_id: int, prefix: str):
        self.module = module
        self.store = store
        self.layer_id = layer_id
        self.prefix = prefix
        self.device = "cpu"
        self._stream_flat = None
        self._flat_offsets = None
        self._cpu_refs = None
        self._flat_order = None

    # -- the contract the engine streams through ---------------------------
    def _iter_tensors(self):
        """(owner, attr, tensor) for every tensor of this layer, in shard order."""
        out = []
        for short, full in self._short_names().items():
            parent_name, _, leaf = short.rpartition(".")
            try:
                parent = self.module.get_submodule(parent_name) if parent_name else self.module
            except AttributeError:
                continue
            if hasattr(parent, leaf):
                t = getattr(parent, leaf)
                if isinstance(t, torch.Tensor):
                    out.append((parent, leaf, t))
        return out

    def _short_names(self) -> dict[str, str]:
        return {n[len(self.prefix):]: n for n in self.store.weight_map
                if n.startswith(self.prefix)}

    def _weight_paths(self):
        return self._iter_tensors()

    def _ensure_flat(self, device):
        """Pack this layer's tensors into one pinned flat buffer plus one device buffer.

        Delegates to the same alignment rule the engine uses for DeepSeek (pad every
        tensor to 256B so the kernels see aligned pointers).
        """
        if self._stream_flat is not None:
            return
        paths = self._weight_paths()
        refs = [t for (_, _, t) in paths]
        order = sorted(range(len(refs)), key=lambda j: -refs[j].element_size())
        offsets: dict[int, tuple[int, int]] = {}
        off = 0
        for j in order:
            t = refs[j]
            nbytes = t.numel() * t.element_size()
            off = (off + 255) // 256 * 256
            offsets[j] = (off, nbytes)
            off += nbytes
        cpu_flat = torch.empty(off, dtype=torch.uint8, pin_memory=True)
        keep = []
        for j, t in enumerate(refs):
            o, n = offsets[j]
            cpu_flat[o:o + n].copy_(t.reshape(-1).view(torch.uint8))
            keep.append(t)
        gpu_flat = torch.empty(off, dtype=torch.uint8, device=device)
        self._stream_flat = (cpu_flat, gpu_flat)
        self._flat_offsets = offsets
        self._flat_order = order
        self._cpu_refs = keep

    def to(self, device: str):
        """Move this layer's weights: one H2D of the packed buffer, views afterwards."""
        if device == self.device:
            return self
        if device == "cpu":
            # swap the original tensors back; zero copy
            if self._cpu_refs is not None:
                holders = getattr(self, "_holders", None) or []
                for (owner, attr, _t), t in zip(holders, self._cpu_refs):
                    setattr(owner, attr, t)
            self._stream_flat = None
            self.device = "cpu"
            return self
        self.materialize()
        # capture the holder/target list ONCE: after the first setattr the attributes point
        # at device views, so recomputing it would return the wrong tensors
        targets = self._weight_paths()
        self._holders = [(o, a, t) for (o, a, t) in targets]
        self._ensure_flat(device)
        cpu_flat, gpu_flat = self._stream_flat
        gpu_flat.copy_(cpu_flat, non_blocking=True)
        for idx, (owner, attr, t) in enumerate(targets):
            off, nbytes = self._flat_offsets[idx]
            view = gpu_flat[off:off + nbytes].view(t.dtype).view(t.shape)
            setattr(owner, attr, torch.nn.Parameter(view, requires_grad=False)
                    if isinstance(t, torch.nn.Parameter) else view)
        self.device = device
        return self

    def materialize(self) -> None:
        """Read this layer's tensors out of the checkpoint (once) into the meta module."""
        if getattr(self, "_loaded", False):
            return
        import torch.nn as nn

        own = dict(self.module.named_parameters())
        own.update(self.module.named_buffers())
        for short, full in self._short_names().items():
            target = own.get(short)
            if target is None:
                continue
            tensor = self.store.get(full)
            if target.is_floating_point() and tensor.dtype != target.dtype:
                tensor = tensor.to(target.dtype)
            parent_name, _, leaf = short.rpartition(".")
            parent = self.module.get_submodule(parent_name) if parent_name else self.module
            if isinstance(target, nn.Parameter):
                setattr(parent, leaf, nn.Parameter(tensor, requires_grad=False))
            else:
                setattr(parent, leaf, tensor)
        self._loaded = True

    def forward(self, hidden, cache=None, position=None, position_embeddings=None,
                attention_mask=None):
        if not getattr(self, "_loaded", False):
            self.materialize()
        out = self.module(hidden_states=hidden, past_key_values=cache,
                          attention_mask=attention_mask, position_ids=position,
                          position_embeddings=position_embeddings, use_cache=True)
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
        self._norm_weight = None
        self._lock = threading.Lock()
        # how many layers may hold real weights at once; the engine sets this from what
        # the machine can actually keep (a 27B layer is 731MB of host memory)
        # how many materialised layers the host keeps; set from the plan's RAM budget
        self._resident_layers = 0
        self._order: list[int] = []

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
        """The layer for this id, materialised on demand and bounded in number.

        The cache is deliberately small and evicts the least recently used entry. A 27B
        layer is 731MB of host memory once materialised, so keeping every layer would need
        47GB for them alone -- the opposite of what streaming is for. Evicting drops the
        module; the next use reads the layer back from the checkpoint.
        """
        # One QwenLayer per id for the process lifetime. The layer object itself is small;
        # what is expensive is its materialised weights, and those are held in ONE flat
        # device buffer that to() reuses. Building a fresh object per call would re-pack
        # that buffer on every token (measured: 47 tok/s -> 2.6 tok/s).
        layer = self._layers.get(layer_id)
        if layer is None:
            # Before building a new one, give room back: a materialised 27B layer holds
            # 731MB of real weights, so the ceiling is what the host can actually keep.
            # Evicting drops the module back to the meta device, and the next use re-reads
            # it from the checkpoint -- which is why the ceiling matters more than the
            # cache hit rate: 64 layers would need 47GB of RAM, and there is 30.
            self._make_room()
            module = self._skeleton().model.language_model.layers[layer_id]
            prefix = f"model.language_model.layers.{layer_id}."
            layer = self._layers[layer_id] = QwenLayer(module, self.store, layer_id, prefix)
        self._order.append(layer_id)
        return layer

    def per_layer_bytes(self) -> int:
        """Bytes one materialised layer holds, from the checkpoint's own tensors."""
        cache = getattr(self, "_per_layer_bytes", None)
        if cache is not None:
            return cache
        total = 0
        headers: dict[str, dict] = {}
        for name in self.store.weight_map:
            if _layer_belongs(name):
                shard = self.store.weight_map[name]
                h = headers.get(shard)
                if h is None:
                    parsed = self.store._header(shard)
                    h = parsed[0] if isinstance(parsed, tuple) else parsed
                    headers[shard] = h
                ent = h.get(name)
                if ent is not None:
                    total += ent["data_offsets"][1] - ent["data_offsets"][0]
                if len(headers) > 2:
                    headers.clear()
        self._per_layer_bytes = total // max(1, self.n_layers)
        return self._per_layer_bytes

    def set_ram_budget(self, gb: float, per_layer_gb: float) -> None:
        """How many materialised layers the host can hold, from a byte budget."""
        if per_layer_gb <= 0:
            self._resident_layers = 0
            return
        self._resident_layers = max(1, int(gb / per_layer_gb))

    def _make_room(self) -> None:
        cap = getattr(self, "_resident_layers", 0)
        if not cap:
            return
        live = [i for i in dict.fromkeys(self._order) if getattr(
            self._layers.get(i), "_loaded", False)]
        while len(live) >= cap:
            victim = live.pop(0)
            layer = self._layers.pop(victim, None)
            if layer is None:
                continue
            module = self._skeleton().model.language_model.layers[victim]
            for name, child in list(module.named_children()):
                try:
                    child.to("meta")
                except Exception:
                    pass
            layer._loaded = False
            layer._stream_flat = None
            layer._cpu_refs = None

    def _touch(self, layer_id: int) -> None:
        order = getattr(self, "_order", None)
        if order is None:
            order = self._order = []
        if layer_id in order:
            order.remove(layer_id)
        order.append(layer_id)

    def _evict_one(self) -> None:
        order = getattr(self, "_order", [])
        for layer_id in list(order):
            if layer_id not in self._layers:
                order.remove(layer_id)
                continue
            self._layers.pop(layer_id)
            order.remove(layer_id)
            # the meta module tree keeps the module object alive; drop the real weights too
            module = self._skeleton().model.language_model.layers[layer_id]
            for name, _ in list(module.named_children()):
                child = getattr(module, name, None)
                if child is not None and hasattr(child, "to"):
                    try:
                        child.to("meta")
                    except Exception:
                        pass
            return

    def forget_layer(self, layer_id: int) -> None:
        """Drop a layer's materialised weights (the caller is done with it)."""
        layer = self._layers.pop(layer_id, None)
        if layer is not None:
            order = getattr(self, "_order", [])
            if layer_id in order:
                order.remove(layer_id)

    def _host_kv(self, kv, device: str):
        """Per-layer KV state on the host, for a layer the engine streamed off the card.

        Each layer owns disjoint slots, so moving one layer's state between devices never
        disturbs another's.
        """
        holder = getattr(self, "_host_cache", None)
        if holder is None:
            holder = self._host_cache = {}
        return _LayerView(kv, device, holder)

    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """The model-level RMSNorm before the head, from the checkpoint."""
        if self._norm_weight is None:
            self._norm_weight = self.store.get(self.norm_key).to(hidden.device).float()
        h = hidden.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True)
                            + float(self.cfg.get("rms_norm_eps", 1e-6)))
        # Qwen scales by (1 + w), not w (transformers PR 29402); using w directly left a
        # scale error that was invisible in the hidden states but wrecked the logits
        return (h * (1.0 + self._norm_weight)).to(hidden.dtype)

    def make_kv(self, capacity: int):
        """A cache with the config attached: without it the layer list stays empty and the
        first linear-attention layer indexes into nothing."""
        from transformers.cache_utils import DynamicCache

        return DynamicCache(config=self._load_hf_config().get_text_config(decoder=True))

    def position_embeddings(self, hidden: torch.Tensor, position: torch.Tensor):
        """Rotary tables for this step, built by the reference implementation.

        RoPE is architecture, not engine: which axes rotate, at what theta, and whether the
        rotation is partial all come from the checkpoint.
        """
        module = self._skeleton().model.language_model
        rotary = module.rotary_emb
        if getattr(rotary, "inv_freq", None) is None or rotary.inv_freq.device.type == "meta":
            # the rotary buffers were built on the meta device; compute them from the
            # config instead of materialising the module (they are 64 floats)
            self._materialise_rope(rotary)
        # mrope splits positions into modality axes: the reference expands to 4 (text plus
        # the three image/video axes), gives layer 0 to the decoder layers and the other
        # three to the rotary tables. A text-only step repeats the same positions in all
        # four, which is what happens here.
        if position.dim() == 3:
            axes = position
        else:
            flat = position.view(1, -1)
            axes = flat.expand(4, -1, -1)
        return rotary(hidden, axes[1:])

    @staticmethod
    def text_positions(position: torch.Tensor) -> torch.Tensor:
        """What a decoder layer receives: 2-D (batch, positions), not the mrope axes."""
        return position[0] if position.dim() == 3 else position.view(1, -1)

    def attention_mask(self, hidden: torch.Tensor, position: torch.Tensor,
                       cache=None, layer_id: int = 0):
        """The mask this architecture needs at this layer, built by the reference code.

        Qwen needs two different ones -- a causal mask for the 16 full-attention layers
        and a recurrent one for the 48 linear layers -- and getting that wrong shows up as
        a shape error deep inside the layer, so it is built here rather than guessed.
        """
        from transformers.masking_utils import (create_causal_mask,
                                                create_recurrent_attention_mask)

        model = self._skeleton().model.language_model
        cfg = model.config
        kind = self.kinds[layer_id]
        # Exactly what the reference does: the same two builders, the same arguments. A
        # None return is meaningful (it tells SDPA to use its own causal path), so it is
        # passed through rather than replaced.
        if kind == "full_attention":
            # Pass through exactly what the reference builds. None is a meaningful value
            # here -- it tells SDPA to use its own causal path -- and substituting a mask
            # changes both the numerics and the shapes.
            return create_causal_mask(config=cfg, inputs_embeds=hidden,
                                      attention_mask=None, past_key_values=cache,
                                      position_ids=position)
        return create_recurrent_attention_mask(config=cfg, inputs_embeds=hidden,
                                               attention_mask=None,
                                               past_key_values=cache,
                                               position_ids=position)

    def _materialise_rope(self, rotary) -> None:
        """Replace the meta inv_freq/attention_scaling with real values from the config."""
        import torch.nn as nn

        cfg = self._load_hf_config().get_text_config()
        inv_freq, scaling = rotary.compute_default_rope_parameters(cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rotary.inv_freq = nn.Buffer(inv_freq.to(device=device, dtype=torch.float32),
                                    persistent=False)
        rotary.original_inv_freq = nn.Buffer(inv_freq.clone().to(device), persistent=False)
        rotary.attention_scaling = float(scaling)

    def make_graph_inputs(self, engine, capacity: int):
        """The device buffers a captured step reads and writes."""
        return (torch.zeros(1, 1, dtype=torch.long, device="cuda"),
                torch.zeros(1, dtype=torch.long, device="cuda"),
                torch.empty(1, engine.cfg.vocab_size, dtype=torch.float32, device="cuda"))

    def capture_step(self, engine, kv, warmup_ids, start_pos: int):
        """Capture one decode step, returning (graph, ids_buf, pos_buf, logits_buf).

        The step allocates nothing: the masks are None (SDPA's causal path), the rotary
        table is a few hundred bytes recomputed in place, and the only writes are the KV
        slots the cache owns. That is what makes the replay legal, and it is what took
        this path from 49.9 tok/s to 180.4 on the 0.8B.
        """
        ids_buf, pos_buf, logits_buf = self.make_graph_inputs(engine, kv.capacity)
        ids_buf.fill_(int(warmup_ids[0, -1]))
        pos_buf.fill_(int(start_pos))

        def step():
            row = self.step(engine, kv, ids_buf, pos_buf)
            logits_buf.copy_(row.reshape(-1).float())

        with torch.inference_mode():
            step()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            torch.cuda.synchronize()
        return graph, ids_buf, pos_buf, logits_buf

    def step(self, engine, kv, ids_buf, pos_buf) -> torch.Tensor:
        """One decode step over the whole model, driven through the engine.

        This is the function the CUDA graph captures: it reads two device tensors and
        writes only the KV buffers, so a replay is legal. Layer residency is decided by
        the engine on each call, which for a fully-resident model means no copies at all.
        """
        import torch.nn.functional as F

        position = pos_buf.to(torch.long)
        # the plan may keep the embedding table on the host (2.4GB for the 27B), so the
        # lookup happens there and the row moves over; for a resident table this is a no-op
        if engine.embed.device.type != "cuda":
            h = F.embedding(ids_buf.to(engine.embed.device), engine.embed).to("cuda")
        else:
            h = F.embedding(ids_buf, engine.embed)
        rope = self.position_embeddings(h, position)
        text_pos = self.text_positions(position)
        for i in range(self.n_layers):
            layer = engine.ensure_layer(i)
            dev = getattr(layer, "device", "cuda")
            # A streamed layer runs on the host (it was never made resident), so the
            # activation, the rotary table and this layer's KV slots follow it and come
            # back. A resident layer touches nothing.
            if dev == "cuda":
                work, pos_i, rope_i, kv_i = h, text_pos, rope, kv
            else:
                work = h.to(dev)
                pos_i = text_pos.to(dev)
                rope_i = tuple(t.to(dev) for t in rope)
                kv_i = _LayerKv(kv, i, dev)
            out = layer.forward(work, cache=kv_i, position=pos_i,
                                position_embeddings=rope_i, attention_mask=None)
            h = out[0] if isinstance(out, tuple) else out
            if dev != "cuda":
                h = h.to(engine.device)
            engine.release_layer(i)
        return F.linear(self.final_norm(h[:, -1:]), engine.head)