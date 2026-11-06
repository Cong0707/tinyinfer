# Measuring the engine without fooling yourself

Four separate times in one session I chased a "bug" that did not exist, and each time the
cause was the measurement rather than the code. They are collected here because the
mistakes are more reusable than the results: the next person (or the next me) will make
the same ones.

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
