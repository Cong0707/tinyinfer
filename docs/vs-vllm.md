# Where this engine wins against vLLM, and where it loses

Both sides measured on the same box (RTX 5060 Laptop, 8GB, sm_120), with vLLM built from
its own 0.33.0 C++/CUDA sources via MSVC + nvcc 13.0 -- not a reimplementation. The
kernel-level numbers are in docs/vllm-baseline.md.

## The model decides the first half of the answer

DeepSeek-V4.1-Flash-1017 is 475.3 GB across 48 shards, of which 296 GB is the routed
expert table. vLLM holds the whole model resident. This engine holds a resident set and
streams the rest.

| hardware | vLLM | this engine |
| --- | --- | --- |
| 8 GB | cannot start (475 GB needed) | 1.26 tok/s, 790 ms/token |
| 20 GB | cannot start | **40/40 layers, 247 slots -- the zero-PCIe threshold** |
| 24 GB | cannot start | 40/40 layers, 309 slots |
| 32 GB | cannot start | 40/40 layers, 561 slots |
| 80 GB | cannot start | 40/40 layers, 2480 slots |
| 141 GB | cannot start | 40/40 layers, 4919 slots |
| 8x141 GB | runs, 42% of VRAM used | runs, same residency, kernel-bound |

The zero-PCIe threshold is 20 GB, not 24. A decode token touches at most topk(6) x 40",
+ "layers = 240 distinct experts at 18.8 MB each, so the slot pool needs 240 entries (4.5 GB)
on top of the 6.5 GB of resident weights and the ring/workspace. Measured with the planner
at each size: 16 GB gives 36/40 layers and no slots at all; 18 GB holds all 40 layers but
only 69 slots (171 short); 20 GB gives 247 slots and every expert of the working set is
already resident.

Below ~475 GB of VRAM there is no comparison to make: one engine runs and the other
does not. That is the whole reason this project exists, and it is worth stating plainly
rather than as a benchmark table with a footnote.

At 20 GB and above the engine is not slower for being a streaming engine -- the planner
fits the entire decode working set (topk 6 x 40 layers = 240 distinct experts) into the
slot pool, so a token crosses PCIe zero times. The 8 GB row is the only one paying for
streaming, and it pays 1.85 GB of expert traffic per token.

### Prefill is where the gap is widest

Everything above is decode. Prefill tells a different story, because every prompt token
passes through every layer's MoE: 51 tokens x 40 layers x 6 experts = 12240 expert visits
against decode's 240. Measured on the 8 GB card:

    prefill 51 tokens   17.3 s   75.75 GB read from disk
    decode 1 token       0.79 s   1.85 GB read from disk

Extrapolating from the same 8.8 GB/s effective read rate, about half of that 17.3 s is
disk; the other half is arithmetic. A 500-token prompt therefore needs roughly 740 GB of
reads, i.e. four to five minutes.

This is the dimension where a resident deployment differs most: on 20 GB or more the
expert table is in VRAM instead of on disk, which takes prefill from ~17 s to ~9 s for
this prompt (-50%) against decode's -36%. It is also where vLLM's 475 GB residency is
worth the most -- it never reads an expert from disk in either phase.

## Where vLLM is genuinely better

The single-blob fp4 expert GEMV, same shape (N=16384, K=5120, group 16, nvfp4, 41.9 MB of
weights), same card, CUDA-event timed:

| M | vLLM Marlin | this engine's GEMV | ratio |
| ---: | ---: | ---: | ---: |
| 1 | 158.3 us (265 GB/s) | 657 us (59 GB/s) | **4.15x slower** |
| 8 | 152.8 us (274 GB/s) | - | - |
| 32 | 190.1 us | - | - |

vLLM's M=1 reaches 265 GB/s against a measured read ceiling of 295 GB/s, i.e. 90% of
what the card can do. Ours reaches 20%. `ncu` attributes the gap to DRAM utilisation 19%,
SM utilisation 52%, and `lg_throttle` 11.37/s: scalar loads, no four-stage cp.async
pipeline, no ldmatrix fragments.

**In a regime where both engines are resident, vLLM would serve this model faster.** The
kernel work needed to close that is understood (adopt the cp.async + ldmatrix structure);
it is not a mystery, just not done.

## What has been closed

| piece | vLLM-equivalent result |
| --- | --- |
| fp8 activation quantiser | bit-identical to the official kernel, including e8m0 scale |
| fp4 quantiser | bit-identical, signed zero preserved |
| fp8 blockwise GEMV | exact match |
| fused gate+up GEMV | 1.50x over the two-call form, both outputs diff 0.0 |
| fp8 tensor-core GEMM | 100% bit-identical to an fp32 reference after bf16 rounding |
| Marlin repack + GEMV | ported from vLLM sources, builds and runs as our own .so |

The correctness gap is closed; the throughput gap on a resident workload is not.

## The honest one-line version

On hardware where vLLM fits the model, it is ~4x faster per GEMV and we have not caught
up. On hardware where it does not fit -- which is every single-card configuration from
8 to 141 GB, i.e. what people actually own -- this engine runs and vLLM does not.
