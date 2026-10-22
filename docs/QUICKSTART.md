# Quickstart: run DeepSeek-V4.1-Flash-1017 on a small box

Everything here is what was measured on an 8GB laptop; the same commands work on bigger
cards (the planner adapts, docs/HARDWARE.md has the table).

## 0. Requirements

- Python 3.11+, torch with CUDA, and this package installed (`pip install -e .`).
- The checkpoint as HF safetensors on a fast disk (510GB). The volume's read speed sets
  decode speed: 4.5GB/s measured best, 1.5GB/s after heavy use.
- ~16GB free host RAM for the pinned slab on an 8GB card (47GB total recommended).

## 1. Generate text (the shortest path)

    python -m tinyinfer.cli generate \
        --backend tiny --model-dir E:\Models\DeepSeek-V4.1-Flash-1017 \
        "Question: What is the capital of France?\nAnswer:" --max-tokens 16

or from Python:

    from transformers import AutoTokenizer
    from tinyinfer.engine import Engine
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eng = Engine.from_hf(MODEL, device="cuda", tokenizer=tok)
    ids = tok.encode("Question: What is pi?\nAnswer:")
    print(tok.decode(eng.generate(ids, max_new_tokens=16)))

Prompt shape matters (base model, no chat template): question/answer framing works,
`<|User|>` markers do not. Details in docs/faq.md.

## 2. Serve it

    python -m tinyinfer.cli serve --backend tiny \
        --model-dir E:\Models\DeepSeek-V4.1-Flash-1017 --port 8000

    curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
         -d '{"prompt": "Question: What is pi?\nAnswer:", "max_tokens": 16}'

OpenAI-compatible: `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/health`.
One request at a time (single KV session).

## 3. Measure it

    python -m tinyinfer.bench --model E:\Models\DeepSeek-V4.1-Flash-1017

JSON with build/first_forward/warm_prefill/decode/hit/disk. Compare `decode_tok_per_s`
against the floor for that run's `disk_gb_s` (docs/PERFORMANCE.md), never against a
number from a different disk state.

## 3b. On an 8-12GB card, use `--tier 1`

Measured on the 8GB box: tier 1 answers the same prompt in 50-69s where the default tier 2
takes 91-94s, and tier 1 has no VRAM slot pool so its logits are reproducible. tier 2's
level-2 features pay off on a card big enough to hold the experts (docs/analysis.md §15).

    python -m tinyinfer.cli generate --backend tiny --tier 1 \
        --model-dir <dir> "Question: What is pi?\nAnswer:" --max-tokens 16

## 3c. What works today (verified)

    Question: What is the capital of France?   ->  Paris
    Question: What is pi?                     ->  Pi is a mathematical constant ... 3.14159
    Question: Who wrote Hamlet?                ->  William Shakespeare

English question/answer framing works; chat markers do not; Chinese prompts land in the
model's exercise/answer-key style (docs/faq.md). A base checkpoint, so short or ambiguous
prompts produce flat distributions and therefore unstable output (docs/analysis.md §20).

## 4. If it is slow

| symptom | first thing to check |
| --- | --- |
| decode much slower than the floor | `disk_gb_s` in the same bench run |
| first run very slow | the 13GB pinned slab setup is one-off |
| answers are off-topic | prompt shape (docs/faq.md), not the engine |
| logits not reproducible | that is the known slot-pool race; `TI_NO_SLOTS=1` (docs/analysis.md §10) |
| prefill slow | `TI_PREFILL_CHUNK` trade-off (docs/analysis.md §13) |

## 5. What the switches do

docs/ENV.md lists every one. The three worth knowing:

- `TI_NO_SLOTS=1` -- deterministic logits, ~40% slower decode.
- `TI_PREFILL_CHUNK=1` -- exactly reference-matching prefill (slower); the planner value
  is the fast path.
- `TI_MOE_PROF=1` / `TI_LAYER_TIME=1` -- diagnostics only, they change nothing.
## 3d. Multi-turn without re-prefilling the conversation

A conversation prefix costs ~43s to prefill and checkpoints to 16.9MB that restores in
~0.01s, so each turn only prefills the new text (verified: population -> 1,000,000, Hamlet
-> Shakespeare, no re-prefill).

    ids = tok.encode(transcript, add_special_tokens=False)
    out = eng.generate(ids, max_new_tokens=16)          # turn 1
    eng.save_checkpoint('/tmp/prefix.pt')

    # next turn
    eng.load_checkpoint('/tmp/prefix.pt')
    new_ids = tok.encode('\nQuestion: ...\nAnswer:', add_special_tokens=False)
    out = eng.generate(new_ids, max_new_tokens=16,
                       start_pos=len(ids), keep_session=True)

`examples/multi_turn.py` is the runnable version. (The HTTP server does not wire this up
yet; it takes one request at a time.)

