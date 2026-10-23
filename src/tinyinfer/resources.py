"""Hardware detection + automatic resource planning for tinyinfer.

Everything that used to be a hand-tuned constant (resident gpu_layers, expert slot
count, RAM cache size, prefill chunk) is derived here from the machine the engine
runs on, so tinyinfer adapts to 8GB laptops, 24GB desktops and unified-memory boxes
without manual flags.

Calibration (measured on RTX 5060 8GB Laptop + 47GB RAM):
  * one resident fp8 layer  : ~0.44 GB VRAM (weights 0.19 + allocator/state)
  * embed (129280x5120 bf16) : 1.32 GB VRAM
  * head  (same shape)       : 1.32 GB VRAM (can stay on CPU)
  * one expert slot          : 18.8 MB VRAM (48 slots ~= one decode token's experts)
  * one streamed layer RVAM : ~0.19 GB host RAM (weights) + experts in the RAM cache
  * practical VRAM ceiling   : physical - 0.6 GB (display/context/FX) - 0.8 GB headroom
"""
from __future__ import annotations

import dataclasses
import os

GB = 1 << 30
MB = 1 << 20

# calibrated model-side constants (fp8 non-expert weights, bf16 embed/head)
LAYER_GB = 0.55           # resident per-layer VRAM incl. allocator/state (recalibrated:
                          # 11 layers + embed + slots measured 8671MB, i.e. 0.55GB/layer)
LAYER_HOST_GB = 0.47      # per-layer host RAM for a streamed layer (measured: the
                          # 34 streamed layers need a 15.9GB pinned slab, 468MB each)
EMBED_GB = 1.32
HEAD_GB = 1.32
EXPERT_MB = 18.8
KV_GB_PER_KTOK = 0.02     # compressed+window KV for 40 layers per 1k tokens (rough)


@dataclasses.dataclass
class Card:
    """One CUDA device as the planner sees it."""
    index: int
    name: str
    total_gb: float
    free_gb: float


@dataclasses.dataclass
class Hardware:
    name: str
    vram_gb: float
    ram_gb: float
    ram_avail_gb: float
    unified: bool = False
    cards: list[Card] = dataclasses.field(default_factory=list)

    @property
    def vram_budget_gb(self) -> float:
        """Total VRAM that may be used across every card."""
        # Use physical capacity, not the snapshot of "free": a leftover CUDA context or
        # the desktop compositor makes free look 1GB smaller and the planner then
        # under-fills the card (0 resident layers on an 8GB card that comfortably runs 2).
        vram = sum(c.total_gb for c in self.cards) if self.cards else self.vram_gb
        if self.unified:
            # One physical pool: the driver's wired-memory limit is ~2/3 of RAM on
            # Apple silicon, and counting RAM *and* VRAM separately double-books the
            # same bytes (a 48GB Mac planned 46GB of VRAM plus 36GB of cache).
            vram = min(vram if vram > 0 else self.ram_gb, self.ram_gb * 0.70)
        # Per-card reserve: display/context/FX, allocator slack and the transient
        # buffers one layer needs while it is being copied in. 0.9GB was measured to be
        # too tight (a planned 6.4GB layout actually allocated 7.0GB on an 8GB card).
        return max(0.0, vram - 1.5 * max(1, len(self.cards) or 1))

    @property
    def ram_budget_gb(self) -> float:
        return max(0.0, self.ram_avail_gb - 4.0)  # leave 4GB for the OS/process


def detect() -> Hardware:
    """Detect the machine and return a Hardware descriptor (never raises).

    All CUDA devices are enumerated: the planner spreads layers over every one of
    them. A device is skipped when it cannot hold the smallest useful unit (one
    layer buffer + its share of slots), e.g. a 2GB card next to an RTX 6000.
    """
    vram_gb = 0.0
    name = "cpu"
    cards: list[Card] = []
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()   # a leftover context must not shrink the plan
            except Exception:
                pass
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                    free_gb = free / GB
                except Exception:
                    free_gb = props.total_memory / GB
                cards.append(Card(index=i, name=props.name,
                                  total_gb=props.total_memory / GB, free_gb=free_gb))
            vram_gb = sum(c.free_gb for c in cards)
            if cards:
                name = cards[0].name if len(cards) == 1 else f"{len(cards)}x {cards[0].name}"
    except Exception:
        cards = []
    ram_gb = ram_avail = 0.0
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_gb = vm.total / GB
        ram_avail = vm.available / GB
    except Exception:
        pass
    return Hardware(name=name, vram_gb=vram_gb, ram_gb=ram_gb, ram_avail_gb=ram_avail,
                    cards=cards)


@dataclasses.dataclass
class Plan:
    gpu_layers: int
    head_on_cpu: bool
    expert_slots: int
    cache_bytes: int
    prefill_chunk: int
    stream_layers: int
    reason: str
    # False when even the streaming skeleton (ring + workspace + tables) does not fit
    # the VRAM budget: the engine refuses to start instead of OOMing mid-prefill.
    fits: bool = True
    # layer -> CUDA index, and per-card expert slot counts. Single-GPU boxes get
    # {layer: 0} and {0: slots} and the engine behaves exactly as before.
    layer_card: dict[int, int] = dataclasses.field(default_factory=dict)
    card_slots: dict[int, int] = dataclasses.field(default_factory=dict)
    embed_on_cpu: bool = False   # 1.32GB of VRAM for a 10KB row lookup: waste when tight
    speculation: bool = False    # the plan actually keeps room for MTP draft stages


def _layer_of(name: str) -> int | None:
    """The layer index a tensor belongs to, whatever the checkpoint's prefix is.

    DeepSeek names layers "layers.N.attn..."; Qwen nests them as
    "model.language_model.layers.N.self_attn...". Both contain a "layers.<int>" segment,
    so find that rather than assuming a prefix.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def model_profile(index: Any) -> dict:
    """Measure what the checkpoint actually costs, per layer and per expert.

    Everything the planner needs comes from here, so it adapts to ANY checkpoint
    instead of a hand-tuned table for this one: layer bytes (non-expert), the largest
    layer (the streaming buffer size), the embedding and head tables, and one expert
    blob. Falls back to calibrated constants only if the index cannot be read.
    """
    prof = {"layer_bytes": 0.19 * GB, "max_layer_bytes": 0.72 * GB,
            "embed_bytes": 1.32 * GB, "head_bytes": 1.32 * GB,
            "expert_bytes": 18.8 * MB, "engram_bytes": 2 * GB, "mtp_bytes": 0, "n_layers": 0}
    try:
        wm = getattr(index, "weight_map", None) or {}
        header = getattr(index, "_header", None)
        if not wm or header is None:
            return prof
        per_layer: dict[int, int] = {}
        for name in wm:
            # Layer tensors are "the ones a layer needs per token". Rather than hardcode a
            # naming scheme (this used to check for attn./ffn.gate./hc_, which is
            # DeepSeek's layout and matched nothing at all in a Qwen checkpoint, so every
            # Qwen layer was costed at DeepSeek's 190MB), derive it from the layer index
            # the checkpoint itself declares.
            lid = _layer_of(name)
            if lid is None:
                continue
            parts = name.split(".")
            if "experts" in parts:
                continue          # routed experts are counted per blob, not per layer
            # Everything else that belongs to a layer IS what the layer streams: the
            # engine asks the layer for its weight paths, so the planner must count the
            # same set. This used to filter on DeepSeek's tensor names (attn./ffn./hc_),
            # which matched nothing in a Qwen checkpoint -- every Qwen layer was costed at
            # DeepSeek's 190MB and the plan then streamed a model that fits entirely.
            try:
                # _header returns (tensor_dict, data_offset), and it raises when a shard is
                # incomplete. Treating the tuple as the dict meant every lookup failed and
                # the whole profile silently stayed at the calibrated defaults -- which
                # happened to be about right for DeepSeek and wildly wrong for anything
                # else (a Qwen layer is ~5MB, not 190MB).
                parsed = header(wm[name])
                tensors = parsed[0] if isinstance(parsed, tuple) else parsed
                ent = tensors.get(name)
                if ent is None:
                    continue
                n = ent["data_offsets"][1] - ent["data_offsets"][0]
            except Exception:
                continue
            per_layer[lid] = per_layer.get(lid, 0) + int(n)
        if per_layer:
            prof["n_layers"] = max(per_layer) + 1
            # The streamed layer is what Block._weight_paths returns. After the RoPE
            # table (256MB) moved to a shared constant it is the fp8 payloads plus a few
            # small bf16 norms, i.e. ~1.0x the header bytes (measured 166MB/layer).
            _m = 1.0
            avg = _m * sum(per_layer.values()) / len(per_layer)
            mx = _m * max(per_layer.values())
            # Two different numbers, used for two different things:
            #   layer_bytes     -> average, for "how many layers fit" (resident * avg)
            #   max_layer_bytes -> worst case, for sizing one flat/prefetch buffer
            # This used to be max(avg, mx), which is just mx: every layer was costed at
            # the worst one. On Qwen3.8-27B (mean 737MB, max 1441MB) that turned a 46GB
            # model into a 92GB plan, so the solver concluded only one layer could be
            # resident, gave the host cache its 256MB floor, and every token then read
            # all 63 streamed layers from disk.
            prof["layer_bytes"] = avg
            prof["max_layer_bytes"] = mx
            # routing width: needed to size the expert pool / decide on coverage
            try:
                import json as _json
                import pathlib as _path
                _cfg = _path.Path(getattr(index, "model_dir", "")) / "config.json"
                if _cfg.exists():
                    prof["topk"] = int(_json.loads(_cfg.read_text(encoding="utf-8"))
                                       .get("num_experts_per_tok") or 0) or None
            except Exception:
                prof["topk"] = None
        for key, name in (("embed_bytes", "embed.weight"), ("head_bytes", "head.weight")):
            try:
                h = header(wm[name])
                enc = h[name]
                prof[key] = int(enc["data_offsets"][1] - enc["data_offsets"][0])
            except Exception:
                pass
        # engram projections are loaded on demand and live beside the layer buffers for
        # the rest of the run; not accounting for them is what made an 8GB card OOM in
        # engram.forward on an otherwise valid plan.
        try:
            eng = 0
            for name in wm:
                # only the small wkv projection stays resident; the giant embedding
                # tables (embed.weight/scale) are row-read from disk and never mapped
                if name.startswith("layers.") and ".engram.wkv" in name:
                    try:
                        h = header(wm[name])
                        ent = h.get(name)
                        if ent is not None:
                            eng += int(ent["data_offsets"][1] - ent["data_offsets"][0])
                    except Exception:
                        continue
            prof["engram_bytes"] = 2.0 * eng     # materialised as bf16
        except Exception:
            prof["engram_bytes"] = 2 * GB
        try:
            prof["expert_bytes"] = int(index.expert_blob_bytes())
        except Exception:
            pass
        # MTP / DSpark draft stages: three small decoders plus the markov head. Only
        # counted when the tier enables speculation, because they cost VRAM that the
        # resident layers would otherwise use.
        try:
            mtp = 0
            for name in wm:
                # routed experts inside the draft stages stream exactly like the main
                # model's (they are not resident): only attn/hc/markov are budgeted
                if name.startswith("mtp.") and ".ffn.experts." not in name:
                    try:
                        h = header(wm[name])
                        ent = h.get(name)
                        if ent is not None:
                            mtp += int(ent["data_offsets"][1] - ent["data_offsets"][0])
                    except Exception:
                        continue
            prof["mtp_bytes"] = 2.0 * mtp
        except Exception:
            prof["mtp_bytes"] = 0
    except Exception:
        pass
    return prof


def _spread_cards(cards: list[Card], n_layers: int, total_resident: int,
                  total_slots: int, ws_gb: float, head_gpu: bool,
                  layer_b: float = 0.19 * GB, max_layer_b: float = 0.72 * GB,
                  embed_b: float = 1.32 * GB, head_b: float = 1.32 * GB):
    """Distribute resident layers and expert slots across every CUDA device."""
    usable = {c.index: max(0.0, c.total_gb - 1.2) for c in cards}
    order = [cards[0].index] + [c.index for c in cards[1:]]
    caps: dict[int, int] = {}
    for idx in order:
        overhead = ws_gb + ((embed_b + (head_b if head_gpu else 0.0)) / GB if idx == order[0] else 0.0)
        caps[idx] = max(0, int((usable[idx] - overhead) / max(layer_b / GB, 1e-6)))
    total_cap = sum(caps.values()) or 1
    layer_card: dict[int, int] = {}
    assigned = 0
    for idx in order:
        want = min(caps[idx], max(0, int(round(total_resident * caps[idx] / total_cap))))
        for _ in range(want):
            if assigned >= n_layers:
                break
            layer_card[assigned] = idx
            assigned += 1
    # leftovers stream through the first card (its flat buffers are already there)
    while assigned < n_layers:
        layer_card[assigned] = order[0]
        assigned += 1
    assigned_per_card = {idx: 0 for idx in order}
    for _i, _c in layer_card.items():
        assigned_per_card[_c] = assigned_per_card.get(_c, 0) + 1
    spare = {}
    for idx in order:
        overhead = ws_gb + ((embed_b + (head_b if head_gpu else 0.0)) / GB if idx == order[0] else 0.0)
        spare[idx] = max(0.0, usable[idx] - overhead - assigned_per_card[idx] * (layer_b / GB))
    tot_spare = sum(spare.values()) or 1.0
    card_slots = {idx: max(0, int(total_slots * spare[idx] / tot_spare)) for idx in order}
    if total_slots and not any(card_slots.values()):
        card_slots[order[0]] = total_slots
    return layer_card, card_slots


def plan(hw: Hardware | None = None, n_layers: int = 40, model: dict | None = None,
         speculation: bool = False) -> Plan:
    """Solve the placement problem from bytes and bandwidths, never from tiers.

    Every number below is a measured price of the hardware (H2D bandwidth, the cost of
    a CPU head) or of the checkpoint (layer/expert/embedding bytes), so a machine that
    does not exist yet still gets a correct plan, and nothing degenerates into "if the
    card is 24GB then ...".
    """
    hw = hw or detect()
    model = model or {}
    layer_b = float(model.get("layer_bytes", 0.19 * GB))
    max_layer_b = float(model.get("max_layer_bytes", 0.72 * GB))
    embed_b = float(model.get("embed_bytes", 1.32 * GB))
    head_b = float(model.get("head_bytes", 1.32 * GB))
    expert_b = float(model.get("expert_bytes", 18.8 * MB))
    n_layers = int(model.get("n_layers", 0) or n_layers)

    V = hw.vram_budget_gb * GB
    cards = hw.cards or [Card(0, hw.name, hw.vram_gb, hw.vram_gb)]
    # No CUDA device at all: everything runs out of host memory (the engine has a CPU
    # path), so a "GPU head" would be a lie. Short-circuit the placement problem.
    has_gpu = bool(hw.cards) and V > 0

    H2D_BPS = 26.5e9            # measured pinned H2D
    PREFILL_TEMP = 0.60 * GB   # dequant/activation peak during prefill
    HEAD_CPU_COST_S = 0.053     # measured: 1.3GB bf16 GEMV on the host
    EMBED_CPU_COST_S = 0.001    # one 10KB row lookup
    # Measured on a live 8GB decode: 4.8GB = ring 2.16 + head 1.32 + 2x0.60 layers
    # + 0.12GB of workspace/KV. The workspace is small at decode; the big figure is
    # only needed transiently by a long prefill, and it is allocated then, not held.
    WS_BASE = 0.20 * GB         # fixed tilelang workspace
    # Measured from a live 8GB run: ring 2.16 + head 1.32 + one layer 0.60 + (workspace
    # + engram + KV) 1.22 = 5.3GB. The workspace scales with the wave but far less than
    # activations do: 0.02GB/token, not 0.15 (which over-reserved 1GB and cost 2 layers).
    WS_PER_TOKEN = 0.02 * GB
    LAYER_SLACK = 1.70          # allocator/state/fragmentation per resident layer,
                                # measured: 355MB of bf16 weights costs ~600MB of VRAM
                                # once its activations and fragments are counted
    SLOT_FRACTION = 0.70
    # ring depth actually allocated by _flat_setup; TI_RING lets the two agree so the
    # planner does not reserve a buffer that is never allocated
    BUF_COUNT = max(2, min(4, int(__import__("os").environ.get("TI_RING", "3"))))

    # VRAM that is gone before a single layer is resident: the on-demand engram
    # projections, the KV window and the activation scratch. Ignoring these is what
    # produced plans that OOM'd in engram.forward / attention on an 8GB card.
    # engram projections are transient (only the engram layers hold them) and KV at
    # short context is small; 1.1GB was over-reserving by ~4 resident layers
    # Prefill runs the whole prompt at once: the fp8 dequant of wo_a (and of the
    # grouped wo matrix) allocates ~128MB per layer on top of the resident set, and
    # the prompt activations add more. Leaving the plan at zero margin put the peak
    # within 128MB of the card's limit, so the first large dequant OOM'd.
    # engram projections are DeepSeek-specific: a model without engram layers must not
    # reserve a gigabyte for them (Qwen reserved 1GB and streamed 5 of its 24 layers for
    # a mechanism it does not have)
    _engram_b = float(model.get("engram_bytes", 0.0) or 0.0)
    runtime_b = (_engram_b * 0.5 + 0.15 * GB + PREFILL_TEMP)
    if speculation and V < 16 * GB:
        # Drafting pays only when the acceptance rate is high; here it is 0.08 (docs/
        # analysis.md section 2), so the draft stages just cost resident layers and the
        # verify pass costs a full forward. Measured on the 8GB card: tier 2 with MTP
        # took 91s for 8 tokens against 50s for tier 1, same answer.
        speculation = False
    if speculation:
        # the draft stages and their windows are resident whenever speculation is on;
        # budget them instead of discovering them with an OOM after prefill
        runtime_b += float(model.get("mtp_bytes", 0.0)) + 0.35 * GB
        # ... but only keep them when the box can still hold the streaming ring plus a
        # couple of resident layers. On an 8GB card the draft stages cost more resident
        # layers than drafting ever repays, so the plan drops them and says so.
        # Drafting only pays if the box can still keep a useful resident set beside it:
        # each resident layer saves a layer-sized H2D per token, and a draft round only
        # adds value when accept > ~1 extra token. Four layers is the measured crossover
        # on this checkpoint (below it, MTP made a 3.0s step into a 3.7s one).
        # the streaming ring is sized by the largest layer in bf16, which measures 2x
        # the header-derived value (719MB vs 355MB on this checkpoint)
        _need = (BUF_COUNT * 2 * max_layer_b + WS_BASE + 8 * WS_PER_TOKEN
                 + float(model.get("engram_bytes", 0.6 * GB)) * 0.5 + 0.15 * GB
                 + float(model.get("mtp_bytes", 0.0)) + 0.35 * GB + 4 * layer_b * LAYER_SLACK)
        if _need > V:
            speculation = False
            runtime_b -= float(model.get("mtp_bytes", 0.0)) + 0.35 * GB

    # A tied checkpoint has no separate head: the head IS the embedding, so reserving
    # head_bytes a second time reserved 1.35GB that is never allocated.
    _tied_head = bool(model.get("tied_head", False))

    def layers_that_fit(embed_gpu, head_gpu, buffers, ws):
        # a tied checkpoint has no separate head allocation (see _tied_head below)
        head_term = head_b if (head_gpu and not _tied_head) else 0.0
        fixed = ((embed_b if embed_gpu else 0.0) + head_term
                 + buffers * max_layer_b + ws + runtime_b)
        return int(max(0.0, V - fixed) // max(1.0, layer_b * LAYER_SLACK))

    ws0 = WS_BASE + 8 * WS_PER_TOKEN
    r_stream = layers_that_fit(True, True, BUF_COUNT, ws0)
    if r_stream >= n_layers:
        embed_gpu, head_gpu, buffers = True, True, 0
    else:
        best = None
        for eg, hg in ((True, True), (False, True), (False, False)):
            r = min(n_layers, layers_that_fit(eg, hg, BUF_COUNT, ws0))
            saved = max(0, r - min(r_stream, n_layers)) * layer_b / H2D_BPS
            price = (0.0 if eg else EMBED_CPU_COST_S) + (0.0 if hg else HEAD_CPU_COST_S)
            cost = price - saved
            if best is None or cost < best[0]:
                best = (cost, eg, hg, r)
        _, embed_gpu, head_gpu, _r = best
        buffers = BUF_COUNT
    # forced head placement (diagnostics/tuning: keeps 1.3GB of VRAM for resident layers
    # at the price of a host GEMV per token)
    if __import__("os").environ.get("TI_HEAD_CPU") == "1":
        head_gpu = False
    else:
        # _head_logits has no host-GEMV path (it slices the head on the device), so a
        # head_on_cpu plan only shows up later as a device mismatch. Keep the head
        # on-card unless the operator explicitly forces TI_HEAD_CPU.
        head_gpu = True
    # forced residency: overrides the solver so a caller can trade the streaming ring for
    # resident layers (the prefill path has to free them again, see Engine._forward_impl)
    _force_resident = __import__("os").environ.get("TI_RESIDENT")
    resident = min(n_layers, layers_that_fit(embed_gpu, head_gpu, buffers, ws0))
    # A model that fits entirely on the card has nothing to trade: pushing the embedding
    # off it would add a PCIe round trip per token and save memory that is not needed.
    # The cost model above assumes a streamed model, where one more resident layer is
    # worth a host lookup; that trade does not exist when nothing streams.
    if resident >= n_layers and has_gpu:
        if not embed_gpu:
            embed_gpu = True
            resident = min(n_layers, layers_that_fit(embed_gpu, head_gpu, buffers, ws0))
    if _force_resident:
        try:
            resident = max(resident, min(int(_force_resident), n_layers))
        except ValueError:
            pass
    if resident >= n_layers:
        buffers = 0
        resident = n_layers
    if not has_gpu:
        # everything runs from host memory on this box
        embed_gpu = head_gpu = False
        buffers = 0
        resident = 0
    elif resident == 0:
        # Too small for even one resident layer with the tables on the device: give the
        # VRAM back to the streaming ring and keep the expert slots minimal but non-zero
        # (a zero-slot plan allocates a fresh 18.8MB buffer per expert per layer per
        # token, which is the failure mode a 4GB card would hit).
        embed_gpu = head_gpu = False
        buffers = min(buffers, 2)
        resident = min(n_layers, layers_that_fit(False, False, buffers, ws0))

    # VRAM headroom knob: on cards where the plan uses ~85% of physical VRAM the driver
    # can page, which makes identical runs differ (docs/analysis.md 19). Lower residency
    # trades speed for a better chance of reproducibility.
    _cap = __import__("os").environ.get("TI_MAX_RESIDENT")
    if _cap:
        try:
            resident = max(0, min(resident, int(_cap)))
        except ValueError:
            pass

    # A handful of expert slots is worse than one fewer resident layer: without a
    # staging pool every expert allocates a fresh 18.8MB device buffer per layer per
    # token. Keep room for two layers' worth of experts (2 x top-k) before trading the
    # remaining VRAM for residency, which is a constraint, not a tier.
    # Slots are expert staging buffers. A model with no routed experts (Qwen, dense
    # models) must not reserve any: holding 12 x 18.8MB out of an 8GB card streamed 7 of
    # Qwen's 24 layers for nothing, and the model fits entirely.
    has_experts = expert_b > 0.5 * MB and float(model.get("topk", 0) or 0) > 0
    MIN_SLOTS = 12 if has_experts else 0
    if not has_experts:
        expert_b = 0.0

    # A tied checkpoint has no separate head: it IS the embedding, so counting both
    # reserved 1.35GB that is never allocated.
    _tied_head = bool(model.get("tied_head", False))

    def _used(r):
        return ((embed_b if embed_gpu else 0.0)
                + (head_b if (head_gpu and not _tied_head) else 0.0)
                + r * layer_b * LAYER_SLACK + buffers * max_layer_b + ws0 + runtime_b)

    used = _used(resident)
    spare = max(0.0, V - used)
    if not _force_resident:
        # the solver only keeps slot room when it chose the residency itself; a caller
        # that asked for a specific count (TI_RESIDENT) wants those layers, not slots
        while resident > 0 and int(spare * SLOT_FRACTION // max(1.0, expert_b)) < MIN_SLOTS:
            resident -= 1
            used = _used(resident)
            spare = max(0.0, V - used)
    if not has_experts:
        slots = 0
    else:
        slots = int(spare * SLOT_FRACTION // max(1.0, expert_b))
        slots = max(slots, 2) if has_gpu else 0

    # ---- expert coverage beats a device-resident table ---------------------------
    # A token can route to topk*n_layers distinct experts. Once the slot pool covers
    # all of them the expert IO term (topk*n_layers*18.8MB per token = 4.5GB, ~1s at
    # 4.5GB/s) disappears; the price of moving the 1.32GB embedding/head table to the
    # host is one 10KB row lookup per token (~1ms). On a 24GB card that is 5 -> 80 tok/s.
    need_slots = int(model.get("topk") or 0) * n_layers

    def _used_with(r, eg, hg):
        return ((embed_b if eg else 0.0) + (head_b if hg else 0.0)
                + r * layer_b * LAYER_SLACK + buffers * max_layer_b + ws0 + runtime_b)

    if need_slots and has_gpu and slots < need_slots * 0.15:
        # Below ~15% coverage the VRAM slot pool costs more than it saves: every expert
        # that is not resident pays a DMA plus an event wait per use, and a slot that is
        # still marked "ready" from a previous use hands back the wrong bytes (measured:
        # 96 slots produced a different first token, and 13 slots made two identical
        # greedy runs disagree). Decode speed comes from the host-pool read path instead.
        slots = 0
    if need_slots and has_gpu and slots < need_slots and (embed_gpu or head_gpu):
        for _eg, _hg in ((False, False), (False, True), (True, False)):
            _r2 = min(n_layers, layers_that_fit(_eg, _hg, buffers, ws0))
            _slots2 = int(max(0.0, V - _used_with(_r2, _eg, _hg)) * SLOT_FRACTION
                          // max(1.0, expert_b))
            if _slots2 >= need_slots:
                embed_gpu, head_gpu, resident, slots = _eg, _hg, _r2, _slots2
                break
    # 8 is the measured sweet spot for this model's tilelang kernel set (a wave of 8
    # needs ~1.6GB of workspace and 165s for a 21-token prompt on 8GB; 16 needs 63s but
    # pages). Larger waves only pay on cards with room to spare.
    chunk = int(max(0.0, spare * (1.0 - SLOT_FRACTION)) // max(1.0, WS_PER_TOKEN))
    chunk = max(8, min(64, chunk))

    # ---- host side: leave the page cache and the CPU tables their memory -----------
    # The expert cache is not free: every GB it takes is a GB the OS cannot use to keep
    # shard data warm, and the measured effect is *negative* past a point (cache 11GB ->
    # 1.45s fetch, cache 15.3GB -> 2.65s fetch on the same 47GB box). Size it from what
    # is genuinely left after the slab, the tables kept on the host, and ~6GB of page
    # cache, then take half of that as the cache.
    host = hw.ram_budget_gb * GB
    if hw.unified:
        # same physical pool: subtract what the device side already booked
        host = max(0.0, host - V)
    # The pinned slab is a PREFETCH buffer, not one slot per streamed layer: the ring is
    # BUF_COUNT deep. Booking (n_layers - resident) * layer_b assumed every streamed layer
    # needs its own pinned copy, which for a dense 27B (63 layers x 1441MB) claimed 88GB
    # of a 47GB box and collapsed the host cache to its 256MB floor -- the model then read
    # every layer from disk on every token.
    slab_b = min(max(0.0, (n_layers - resident) * layer_b),
                 max(1, buffers) * max_layer_b)
    cpu_tables = (embed_b if not embed_gpu else 0.0) + (head_b if not head_gpu else 0.0)
    page_cache_b = min(6.0 * GB, 0.20 * host)
    # 0.8 of the remainder measured best: 4.8GB of cache held only one token's experts
    # (273 entries) so the second token missed everything and fetch doubled; ~10GB
    # reaches the 40% hit rate that keeps the SSD inside its 4.6GB/s envelope.
    cache_b = int(max(256 * MB, (host - slab_b - cpu_tables - page_cache_b) * 0.8))
    cache_b = min(cache_b, 128 << 30)

    if len(cards) <= 1:
        layer_card = {i: 0 for i in range(n_layers)}
        card_slots = {0: slots}
    else:
        layer_card, card_slots = _spread_cards(cards, n_layers, resident, slots,
                                               ws0 / GB, head_gpu, layer_b,
                                               max_layer_b, embed_b, head_b)
    fixed_b = ((embed_b if embed_gpu else 0.0) + (head_b if head_gpu else 0.0)
               + buffers * max_layer_b + ws0 + runtime_b)
    fits = fixed_b <= V
    if not fits:
        need_gb = (fixed_b + 1.5 * max(1, len(cards) or 1) * GB) / GB
        reason_extra = f" INSUFFICIENT VRAM: needs about {need_gb:.1f}GB for the streaming skeleton"
    else:
        reason_extra = ""
    reason = (
        f"{hw.name} vram={hw.vram_gb:.1f}GB({len(cards)} card) ram={hw.ram_gb:.1f}GB"
        f"(avail {hw.ram_avail_gb:.1f}) -> layers={resident}/{n_layers}"
        f" stream={n_layers - resident} buf={buffers} slots={slots}"
        f" cache={cache_b/GB:.1f}GB chunk={chunk}"
        f" embed={'gpu' if embed_gpu else 'cpu'} head={'gpu' if head_gpu else 'cpu'}"
        f" layer={layer_b/MB:.0f}MB expert={expert_b/MB:.1f}MB{reason_extra}"
    )
    return Plan(
        layer_card=layer_card,
        card_slots=card_slots,
        speculation=bool(speculation),
        embed_on_cpu=not embed_gpu,
        gpu_layers=resident,
        head_on_cpu=not head_gpu,
        expert_slots=slots,
        cache_bytes=cache_b,
        prefill_chunk=chunk,
        stream_layers=n_layers - resident,
        reason=reason,
        fits=fits,
    )
