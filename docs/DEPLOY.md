# Deployment

Everything below is verified against this repository at the commit that adds this file;
the numbers are measured, not estimated. Section 8 lists exactly what has NOT been verified.

## 1. What the machine needs

| | minimum | what it decides |
| --- | --- | --- |
| GPU | any CUDA card with >= 8 GB | how many layers stay resident and whether expert slots exist |
| CUDA toolkit | 12.x (`nvcc` on PATH) | compiling the kernels AND tilelang's JIT at runtime |
| host RAM | 16 GB (47 GB recommended) | the pinned slab + expert cache |
| disk | the checkpoint volume's read speed | decode speed, directly |
| python | 3.12+ | venv is created 3.13 |

The FPGA-free rule: **the checkpoint volume's sequential read speed sets decode speed.**
Measured on the reference 8 GB laptop: 4.5 GB/s early, 1.5-1.9 GB/s after heavy use. Same
code, 3x difference in tok/s.

## 2. Checkpoint

HF safetensors of DeepSeek-V4.1-Flash-1017 (about 510 GB), see section 7. Keep it on the
fastest volume. Several directories can be striped:

    ./start.sh --model '/mnt/nvme0/model;/mnt/nvme1/model'

## 3. Install

    git clone <repo> tinyinfer && cd tinyinfer
    ./scripts/bootstrap.sh

bootstrap.sh creates the venv, installs torch for the local driver, installs the package
editable with its dependencies (this includes **tilelang**, which is a hard dependency --
see section 6), and compiles the kernels for the GPU it finds.

torch is not pinned on purpose; if the default wheel does not match the driver:

    TORCH_INDEX=cu130 ./scripts/bootstrap.sh

## 4. Compile the kernels for the target card

`bootstrap.sh` detects the arch, which is right when you build ON the machine that runs it.
Cross-building (or a driver that reports something unexpected) must pass it explicitly:

    ARCH=sm_90  ./scripts/build_native.sh      # H100 / H200
    ARCH=sm_120 ./scripts/build_native.sh      # RTX 50xx
    ARCH=sm_75  ./scripts/build_native.sh      # Turing (T4, RTX 20xx)

Marlin's two translation units need CUTLASS headers:

    CUTLASS_PATH=/path/to/cutlass ARCH=sm_90 ./scripts/build_native.sh

Verified archs in this repository: `sm_75`, `sm_90`, `sm_120` -- all 22 kernels compile.
Without cubins the engine still runs: every dispatch falls back to torch, which is 5-10x
slower but numerically the same.

## 5. Run

    ./start.sh --model /models/DeepSeek-V4.1-Flash-1017                 # foreground
    ./start.sh --model /models/DeepSeek-V4.1-Flash-1017 --bg            # -> serve.log
    TINYINFER_MODEL=/models/... TINYINFER_PORT=8000 ./start.sh

or directly:

    python -m tinyinfer.cli serve --backend tiny \
        --model-dir /models/DeepSeek-V4.1-Flash-1017 --port 8000

OpenAI-compatible: `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/health`.

    curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
         -d '{"prompt": "Question: What is pi?\nAnswer:", "max_tokens": 16}'

### Multi-GPU

Nothing to configure. The engine probes every visible CUDA device at startup, builds a
placement table, and spreads resident layers, expert slots and (if the cut is big enough)
expert ownership across the cards. Restrict it with `CUDA_VISIBLE_DEVICES` if you do not
want all of them used. A single card is the same code path with a one-entry table.

## 6. Read the first line of the log

This one line says what the engine decided and therefore where the bottleneck will be:

    [tinyinfer] <GPU> vram=.. ram=.. -> layers=R/N stream=S buf=.. slots=.. cache=..GB

| field | meaning |
| --- | --- |
| `layers=R/N` | R layers resident on the GPU, N-R re-streamed every token |
| `stream=S` | layers whose weights are re-copied per token (S x 166 MB of H2D) |
| `slots` | VRAM expert staging slots. **0 means every expert visit re-uploads 18.8 MB** |
| `cache` | host RAM expert cache; sets the disk hit rate |

Two reference points, measured with the planner itself:

    8 GB card:   layers=10/40 stream=30 slots=0      -> H2D/disk bound
    141 GB card: layers=40/40 stream=0  slots=4986   -> compute bound (240 needed)

If you expected compute-bound and see `stream>0` or `slots=0`, the plan is wrong for the
machine -- report that line, not a guess.

## 7. Verify and measure

    python -m tinyinfer.bench --model /models/DeepSeek-V4.1-Flash-1017

JSON with `build_s`, `first_forward_s`, `warm_prefill_s`, `decode_step_s[]`,
`decode_tok_per_s`, `expert_cache_hit`, `expert_cache_entries`, `disk_gb_s`.

Compare tok/s only against `disk_gb_s` from the SAME run -- a fetch-bound engine's speed is
a property of the volume as much as of the code.

## 8. What has and has not been verified

Verified:

- 22 kernels compile for `sm_75`, `sm_90`, `sm_120`
- bf16 conversion software path on Turing is bit-exact against torch (512/512)
- end-to-end decode on an 8 GB laptop; output is token-identical across the optimisations
- planner output for 1 x 141 GB and 4 x 141 GB

NOT verified:

- **any kernel has executed on SM90.** Compiling is not running; the `caps.py` gates open
  `tma` / `warp_specialized` on SM90 by capability, and that path has never been exercised.
- multi-GPU execution (the machine that produced these numbers has one card). Placement,
  expert routing and partial-sum reduction are written and unit-tested, not run.
- CUDA graph replay beyond a single position: capture requires `gpu_layers == max_layers`
  (so it never triggers on a streaming box), and `_graph_run` only replays at the captured
  `start_pos`. Expect eager decode on every other step.

## 9. Common failures

| symptom | cause |
| --- | --- |
| `ImportError: ... tilelang` during prefill | tilelang missing; install the project deps |
| `mat2 is on cpu` | a weight did not get swapped by flat streaming; open an issue with the planner line |
| decode much slower than the log suggests | `disk_gb_s` in the same bench run dropped; check the volume |
| `CUDA graph unavailable (...)` | expected on a streaming plan; informational |
| OOM at startup | the plan assumed more VRAM than exists; `TI_MAX_RESIDENT=<n>` forces fewer resident layers |
