# Measuring the engine without fooling yourself

Four separate times in one session I chased a "bug" that did not exist, and each time the
cause was the measurement rather than the code. They are collected here because the
mistakes are more reusable than the results: the next person (or the next me) will make
the same ones.

## 0. A multi-threaded counter is not a wall clock

This one cost more time than the other four combined. `ShardReader` sums
`read_seconds += time.perf_counter() - started` inside every worker, so its total is the
sum of per-thread durations, not elapsed time. Over six decode steps it accumulated
4057 ms while those steps took 4300 ms in total -- if that number were treated as wall
time it would report a 94696% share of a token.

Consequence: two separate conclusions were drawn from it and both were wrong.

  - "disk is 85% of a decode step" -- computed as 2.49 GB / 3.7 GB/s, assuming serial
    reads that never happen (six workers are in flight).
  - "the disk is fully hidden" -- inferred from a cache-size change that cut bytes 18%
    without changing token time, which only shows the change was too small to cross a
    threshold, not that the term is free.

The correct measurement wraps `perf_counter` around the *calls* on the calling thread:

    fetch (ExpertStore.fetch, blocks on the async read)   285 ms   36%
    gate                                                   36 ms    4%
    everything else                                       474 ms   60%

and the regression over twelve tokens confirms it is real: `corr(fetch_ms, disk_MB) =
0.9986`, `fetch_ms = 114 us/MB * disk_MB + 74 ms`.

**Rule:** a counter that sums worker durations answers "how much work did the threads
do", never "how long did the caller wait". For the latter, time the call.

## 1. The benchmark function had a side effect

`Engine.stream_generate(prompt, max_new_tokens=1)` calls `reset()` and then forwards the
whole prompt. Calling it in a loop to time "one token" actually times

    prefill(prompt_len) + 1 decode token

for every iteration, so the measurement grew linearly with prompt length and looked like
a 10x regression at 11 tokens. The engine was fine: one prefill followed by repeated
`forward(one_token, start_pos=pos)` gives 678 ms at 1 token and 964 ms at 11, which is
the expected KV-cache cost.

**Rule:** time the primitive (`forward`), not the entry point that wraps it in a session
reset. If the thing under test has a side effect, either neutralise it or measure below it.

## 2. An exception in cleanup hid three failing assertions

`loading.selftest()` raised `PermissionError` from `TemporaryDirectory.cleanup()` on
Windows, inside a `finally:` block. Because that fired before the assertions in the body
ran, the selftest had been failing silently in a way that looked like a temp-directory
problem. Making cleanup tolerant of a locked file exposed three genuinely stale asserts
(an async prefetch counted as if it were synchronous, and a slot's bytes expected to
vanish after release when the whole point of the pool is that they stay).

**Rule:** a `finally:` that can raise will mask whatever the body was doing. Swallow in
cleanup, and let the body speak.

## 3. An open-ended prompt made a correct answer look like an echo

`/v1/chat/completions` with `"The capital of France is"` returned
`"The capital of France is"`, which looks exactly like "the server echoes its input". It
is not: that sentence is an open completion, and with only 10 of 40 layers resident the
model's greedy decode degrades into repeating its own prefix. Asking something with one
right answer -- `"Answer in one word. The capital of France is"` -- returns `Paris` from
both endpoints.

**Rule:** an end-to-end test needs a question with a unique correct answer. An open
prompt's legitimate response can be the exact string you are about to call a bug.

## 4. `data_ptr` is not a stable identity

`gemv_fp4_marlin._prepared` caches the repacked Marlin weight by `w_fp4.data_ptr()`. An
expert's fp4 tensors are views into the 4-deep `_blob_stage` ring, so the pointer rotates
and the cache never hits: profiling one decode step showed 2217 repack launches against
~240 experts. I "fixed" it by reordering the dispatch, which made things slower, and
reverted.

**Rule:** cache keys must be identities that outlive the buffer -- an expert id, not an
address into a recycled ring. (Still open: the ordering is a real issue on top of the
keying, and needs a fix that does not also drop the fused path.)

## Checking a change here

    python -m pytest tests -q --ignore=tests/test_server.py   # 50 cases
    python -c "import sys; sys.path.insert(0,'src'); from tinyinfer.loading import selftest; assert selftest()"
    python scripts/check_kernels.py                          # 78 conformance cases
    python scripts/predict.py --model <dir>                  # planner table

The decode reference is `scripts/bench.py`; run it standalone, not inside another
process, and read `decode_tok_per_s_steady` rather than the mean.
## Dead ends, with the measurement that closed them

Everything below was implemented far enough to measure, and then reverted. They are kept
because 'we tried X and it did not work' is only useful with the number that killed it.

| attempt | measured result |
| --- | --- |
| cross-layer router prefetch | 790 -> 1216 ms. The next layer's router can be evaluated
  from the current layer's output (0.052 ms each), but its prediction agrees with the real
  routing at Jaccard 0.150, so 85% of what gets fetched is the wrong expert and the disk
  sees more traffic, not less. |
| marlin repack cache | 1.9 s -> 4.8 s (64 entries) -> 11.3 s (512). Retaining a repacked
  expert also pins its source tensors (6 MB each); on a 6.9 GB card that evicts buffers
  the streaming ring needs. Unbounded it trips the CUDA allocator. |
| more reader threads | 8/16/32 threads: 2.88/2.99/2.89 GB/s. Saturated at 8. |
| reading through the Windows file cache | 1.06 GB/s against 3.0 with FILE_FLAG_NO_BUFFERING
  (decode 726 -> 1103 ms). |
| larger expert cache | 15.8 -> 27.5 GB buys hit rate 49.3% -> 52.5%, i.e. ~6% of a token,
  for 11.7 GB more RAM on a 47.2 GB box. |
| coalescing a layer's 28 DMAs into 3 | needs the host addresses to be contiguous; on
  Windows cudaHostAlloc scatters them over 2.4 TB of address space, so the path cannot
  engage. Works on Linux. |
| tensor-core GEMM at M=1 | 4.2x (wq_b) to 6.5x (w1) slower than the SIMT GEMV; 15/16 of
  the 16-row tile is padding. |
| CPU-side MoE routing | 788 -> 879 ms; the D2H sync costs more than the matmul it saves. |

## One more measurement trap worth naming

Deleting a code block by line-number slice corrupted `engine.py`: the offsets were read
from one snapshot of the file and applied to another, so a non-adjacent range went with
it. `git checkout --` recovered it. Use an edit with a unique anchor; line numbers are not
stable across a read and a write.
