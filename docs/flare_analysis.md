# NVIDIA Flare — what to take, and where it pays

`repositories/flare-local-inference` (cloned). NVIDIA's local inference server for
**SM120/SM121** — the same GPU family as our RTX 5060. Architecture: Python *Ignition*
prepares a model (PyTorch + FlashInfer/b12x + `torch.compile`), captures CUDA graphs,
exports a `.flare` package; the Rust *Helios* runtime serves it with no Python, no
runtime compilation and no steady-state GPU allocation. Model recipes are narrow
(Qwen/GLM) but **`ignition/recipes/deepseek41/dspark.py` covers our checkpoint's MTP**.

It cannot replace tinyinfer (it only loads its own prepared packages, and has no
DeepSeek-V4.1 recipe for the 510GB MoE weights), but several mechanisms transfer.

## 1. CUDA graphs as the execution boundary  ★★★

Flare captures a **catalog** of graphs: prefill buckets {128,256,512,1024,2048,4096},
decode batches {1,2,4}, and every speculative width — 25 graphs plain, 40 with five
drafts. Serving selects a graph; nothing launches eagerly.

We launch ~2240 tilelang kernels per decode step through `tvm_ffi`. Our profile shows
`serve` 130–170ms and part of `bincount` 300–470ms are host submission/sync cost.

| hardware | value |
| --- | --- |
| 5060 8GB | **High**: the flat ring gives stable addresses (3 buffers, `k % 3`), so 3 decode graphs cover the non-MoE segment. Est. 300–500ms/step (2.95s → ~2.5s). |
| 5090 32GB | **Higher**: batch>1 requires the catalog anyway; graphs are the difference between launch-bound and compute-bound. |
| 6000 Pro 96GB | **Mandatory** for multi-request serving. |

MoE cannot go in a graph (expert slots are dynamic); attn/hc/gate can.

## 2. Keep the sampled token on the device  ★★★

Flare's replay loop keeps pending token, position, RNG and penalty histogram in
**persistent device storage per request slot**; the decode graph writes the sampled
token back and the next decode consumes it. Two completion frames, at most two steps
in flight; a copy stream moves only token/status records.

We do: `logits` on GPU → `.argmax().item()` (a full sync) → new input tensor → H2D.
That is one hard synchronization per token, on every machine. Value is universal,
implementation is small (device-side argmax + a persistent input buffer).

## 3. MTP feedback pairs — likely why our accept is 25%  ★★★

Flare's DSpark section is explicit: the draft layers' **128-token window holds
`consumed_tokens` committed `(hidden, token)` pairs**, refreshed by a dedicated
`b12x.sequence.mtp_feedback` operator ("four streams; caller-owned output and scratch").
Every commit **absorbs the accepted target feedback into the draft windows**, and
"repair draft history using actual target residuals". Prefill has its own
`prefill+mtp_absorb` graph; resume has `resume+mtp_seed`.

Our `engine._draft` calls `st.attn.reset()` before **every** draft, so the draft sees
only the current token's target hidden — no history, no feedback pairs. Our Markov head
is already correct (autoregressive on the previously sampled draft token), and the seed
path (target inputs to layers 37/38/39, collapsed mHC, concatenated) matches. The
window is the difference.

| hardware | value |
| --- | --- |
| 5060 8GB | Negative today: MTP needs ~1.5GB VRAM we do not have, and accept 25% costs more than it saves. |
| 5090 32GB | **High**: a correct feedback window is the difference between 25% and the 60–75% draft-1 agreement we measure; 2–3x effective throughput. |
| 6000 Pro | Same, plus adaptive width selection (Flare picks among captured widths from observed acceptance and measured round cost). |

## 4. Registered host staging + tiered caches  ★★

`prefix-cache-offload.md`: CompositeStore (registered memory + write-through to disk) /
MemoryStore / DiskStore, a **bounded registered staging pool**, reads that warm the
memory tier, and "offload never blocks the model worker thread or the model stream".
`windows.md`: disk embedding tables use **registered files and I/O rings**, four
workers sharing 128 slots and 8MiB staging, unbuffered overlapped I/O with bounded
queues, 4KiB alignment enforced.

We have the same shape (pinned slab, ExpertCache, `_WindowsDirect` unbuffered reads,
recycled pinned blob pool) but nothing is registered with the driver and our fetch
blocks the caller.

| hardware | value |
| --- | --- |
| 5060 8GB | **Medium**: registration could lift our 3.7–4.6GB/s expert reads; async fetch is the bigger win (fetch is 1.5–2.9s of a 2.95s step). |
| 5090 32GB | Lower — most experts fit in VRAM, the disk path matters less. |
| 6000 Pro | Low for weights; still relevant for prefix-cache spill. |

## 5. Immutable prefix checkpoints  ★★

A checkpoint = KV pages + every continuation state (recurrent matrices, convolution
history, selectors, sliding window, last hidden) + draft windows, published at message,
prompt and terminal endpoints; branches share pages and own private suffixes. Resume
sampling replays only the vocabulary projection from the saved final hidden.

We have none: every request recomputes. On 8GB it is not affordable (the cache budget
is already split three ways). On 32/96GB with agent-style repeated prefixes it removes
the entire prefill cost for a repeated conversation.

## 6. Same-process tensor parallelism  ★★ (large cards only)

Ranks as threads of one process, one CUDA context each, VMM arenas
(`cuMemCreate/cuMemMap/cuMemSetAccess`, 1TiB reservations, peer access per mapping),
PCIe barriers via system-scope stores. No IPC handles.

We planned *layer* splitting across cards, which is pipeline parallelism: it adds a
hop per layer boundary but needs no collectives. Flare's TP splits every GEMM across
ranks, which is better for latency on NVLink-class or PCIe-5 cards but needs
all-reduce per layer. For 2×5090/6000Pro, TP is the better target; our layer split
remains useful for heterogeneous pairs (e.g. 8GB laptop + 96GB external).

## 7. Scheduling as graph selection  ★★ (≥2 concurrent requests)

A graph is a **nonpreemptible service quantum**; the scheduler picks quanta and lanes
by useful work and service value (EEVDF-style accounting, continuation value, adaptive
speculation width from observed acceptance). Irrelevant to us today (one token at a
time), but any server mode on 32GB+ needs it — and it is the reason their per-token
latency is stable under load.

## 8. Smaller but real

- **`--mtp-draft-tokens 0`** keeps the draft weights/history but serves width-one
  graphs so drafting can resume: a clean way to make speculation optional per request.
- **Capacity buckets + "padding changes capacity, never logical length"**: padded rows
  must not write KV or commit state. Our prefill chunks pad; the same rule applies.
- **Graph timings learned online** (hidden durations, completion observations) — they
  model estimation error instead of assuming a measured table.
- **Compressed scale layout** that survives TP1/2/4 without repacking — needed only
  if we ever shard our checkpoint.
- **`vocabulary projection from a saved hidden vector`** (resume sampling graph): a
  cheap way to re-sample without a full forward.

## Verdict by machine

| machine | take |
| --- | --- |
| RTX 5060 8GB (now) | CUDA graph for the non-MoE segment; device-side token handoff; async expert fetch. Skip MTP (no VRAM). |
| RTX 5090 32GB | + MTP feedback window (biggest single win), graph catalog for batch, prefix checkpoints. |
| RTX 6000 Pro 96GB | + same-process TP, compressed scales, scheduler, full graph catalog. |

The one thing Flare does that we should not copy: requiring a prepared package. Loading
the HF checkpoint directly is why this engine exists, and it costs us nothing when the
planner is right — which is the work still in progress.
