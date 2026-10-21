# Extending tinyinfer

## Adding a model family

`adapters.py` defines the contract: `load_config`, `has_tensor`, `get_tensor`,
`build_block`. The DeepSeek V4.1 adapter is the first implementation; a GGUF-based
adapter is reserved. A new adapter must provide a `TinyConfig` and blocks that expose
`forward(x, start_pos, pre_mix)` and `_weight_paths()` (the streaming planner reads the
latter to size the slab).

## Adding an operator

1. Put the reference in `probe_*.py` and record the max relative difference.
2. Write it in TileLang under `deepseek_kernels.py` (or a new module) with an interface
   that takes caller-owned output tensors when the dtype is fp8/e8m0 (torch cannot
   convert those through DLPack).
3. Verify bit-exactness or document the residual; gate it behind a tier flag or env
   switch if it is not exact.
4. Wire it into the hot path only after parity passes.

## Adding a service endpoint

`server.py` (FastAPI) already covers the OpenAI surface: `/v1/models`,
`/v1/completions`, `/v1/chat/completions`, `/health`, SSE streaming, API keys and
per-session slots. New endpoints go there; `cli.py serve` runs it.

## Changing the planner

`resources.py` derives residency, ring, cache and chunk from the machine. If a zero-
config run misbehaves, fix the planner or the adapter rather than adding a flag: the
project's rule is that flags are for measurement, not for making it work.

## What not to add

- CUDA C++, CuTe DSL, Triton or vendor binaries (portability is the point).
- Approximations below level 3.
- Machine-bound constants; read the hardware instead.
