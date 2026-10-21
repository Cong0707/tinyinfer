# tinyinfer

A local inference engine for DeepSeek-V4.1-Flash-1017 (510GB, 40 layers, 384 routed
experts top-6, fp8 non-expert + fp4 experts) that runs **on an 8GB laptop GPU with
47GB of RAM** by streaming layers and routed experts from the checkpoint on disk.

## What works

- Full model load straight from the HF safetensors (no conversion step).
- Streaming decode and prefill; resident set is 4-5 layers (the card holds no more).
- Multi-session reuse: `generate()` starts from a clean session every time.
- HTTP server (`python -m tinyinfer.serve`) with `/health` and `/generate`.
- Quantisation levels 0-3 (`OptLevel`): 0 reference / 1 production-proven ports
  (official TileLang ops, NVIDIA Flare mechanisms) / 2 engine features (streaming,
  expert cache, gated prefetch, MTP stages) / 3 approximations. **Level 2 is the
  minimum for this machine.**
- Numerics verified against DeepInfer (the model vendor's own stack) where it can run:
  window-layer attention 0.000000, fp4 expert 0.000000, gate 0.000000, hyper-connection
  ops 0.000000, RoPE 9.7e-06, compressed layers 0.87%/1.42%.

## Measured on this box

| metric | value |
| --- | --- |
| decode | **1.66s/token = 0.60 tok/s** at 4.5GB/s disk; 0.4-0.5 tok/s at the 1.4-1.9GB/s the volume currently reports |
| theoretical floor | 1.23s/token = 0.81 tok/s (IO only) |
| prefill, 8 tokens, warm | ~3s |
| process start (pinned slab, 13GB) | ~13s, one-off |
| output | `Question: What is the capital of France?\nAnswer: -> Paris`, and `What is pi? -> 3.14` (see docs/faq.md for prompt shapes) |

## Hardware reality

The IO floor (docs/lower_bounds.md) is computed from measured constants: expert
blobs 18.8MB at 4.5GB/s and a 166MB streamed layer per token. 1 tok/s needs fewer
routed experts (quality loss) or more VRAM; top-4 measured 1.36s/token with visibly
degraded output.

## Usage

    export E:\Develop\Deepseek\tinyinfer\.venv\Scripts\python.exe
    export PYTHONPATH=E:\Develop\Deepseek\tinyinfer\src

    python -m tinyinfer.cli serve --backend tiny --model-dir E:\Models\DeepSeek-V4.1-Flash-1017 --port 8000
    curl -X POST http://127.0.0.1:8000/generate -d '{"prompt": "...", "max_new_tokens": 8}'

    # the base model answers question-shaped prompts best (measured):
    #   "Question: What is the capital of France?\nAnswer:" -> " France is a country in Europe..."
    #   "Question: What is pi?\nAnswer:"                   -> " The answer is 3.14..."
    #   multi-turn: append 'Question: ...\nAnswer: ...' pairs to one transcript

    # or from Python
    from tinyinfer.engine import Engine
    eng = Engine.from_hf(model_dir, device="cuda", tokenizer=tok, allow_missing_layers=True)
    out = eng.generate(token_ids, max_new_tokens=8)

    # examples and tooling
    python examples/generate.py --model <dir> --prompt "The capital of France is"
    python examples/stream.py   --model <dir> --prompt "Once upon a time"
    python examples/tiers.py    --model <dir>
    python -m tinyinfer.bench   --model <dir>        # JSON metrics (prefill/decode/cache/disk)

Zero configuration is the supported default: the planner reads VRAM/RAM/disk and
picks resident layers, ring size, cache size and chunking itself.

## Docs

- docs/lower_bounds.md  - what the hardware physically allows (arithmetic)
- docs/decode_speed.md  - measured steps and what each optimisation changed
- docs/parity.md        - numeric parity against DeepInfer per component
- docs/kernels_provenance.md - every kernel: official port / re-implementation
- docs/level2_mechanisms.md  - the level 1/2 mechanisms and where they come from
- docs/flare_analysis.md - NVIDIA Flare (b12x) mechanisms, ported and rejected ones
- docs/benchmarks.md   - benchmark history (the disk rate is part of every row)
- docs/PERFORMANCE.md  - one-page performance summary and how to measure it
- docs/analysis.md     - routing predictability, speculative ceiling, batch-vs-row probe
- docs/architecture.md - module map, streaming model, expert path, sessions
- docs/ENV.md          - every environment switch
- docs/ROADMAP.md      - next steps ordered by measured value, with measured non-goals
- docs/faq.md          - the questions this design invites