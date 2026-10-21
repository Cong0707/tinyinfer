# Operations

## Start

    PYTHONPATH=src python -m tinyinfer.cli serve --backend tiny \
        --model-dir E:\Models\DeepSeek-V4.1-Flash-1017 --port 8000

First start pays a one-off ~13s pinned-slab setup (13GB cudaHostAlloc) after the ~2s
weight index. The server answers `/health` immediately and 503s generation until ready.

## Monitor

- `GET /health` -- readiness and the planner's decision summary.
- `python -m tinyinfer.bench --model <dir>` -- prefill/decode/cache/disk as JSON; run it
  after every deployment and compare against docs/PERFORMANCE.md.
- The engine prints one line at start-up with resident layers, ring size, cache and chunk;
  a change there usually explains a speed change.

## Failure modes seen

| symptom | cause | action |
| --- | --- | --- |
| 503 forever | model still loading | wait; check the server log |
| OOM at first token | another process holds VRAM | free it; the engine sheds the ring and retries once |
| decode much slower than docs | checkpoint volume slow | measure the disk rate with bench; it is fetch-bound |
| generation degenerates to BOS | MTP forced on a card too small | do not set TI_MTP=1 on 8GB |
| off-topic text | base model (no chat alignment) | use question-shaped prompts (docs/faq.md) |

## Resource budget (this box)

| item | size |
| --- | --- |
| pinned layer slab | ~13GB host |
| expert cache | ~15GB host |
| CPU tables (embed) | ~1.3GB host |
| device ring | 3 x 463MB |
| resident layers | 4-5 x 178MB |
| head | ~1.3GB device |

The planner recomputes this from what the machine reports; the numbers above are what it
chose on 8GB/47GB.

## Logs

Start-up prints the hardware plan and the slab mode. `TI_TIME=1` adds per-step decode
timing, `TI_MOE_PROF=1` the MoE phase split; both are diagnostics only and change nothing.
