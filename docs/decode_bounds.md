# What bounds decode speed on an 8 GB card

Every number here was measured on the development box (RTX 5060 Laptop, 6.9 GB usable,
47.2 GB RAM, Windows) with the full 40-layer model. Several earlier versions of this
analysis were wrong and are corrected below; the corrections are the useful part.

## The whole budget

    790 ms/token = 1.26 tok/s       steady state, output [223,19,17,20]

    term                  value        share
    expert bytes / token  2.49 GB
    disk rate (direct)    3.0 GB/s     --
    disk time             830 ms       100% (overlapped with compute)
    launch cost           6.1 ms       0.8%

Decode time is, to first order, `expert_bytes / disk_rate`. Nothing else is close.

## The three ceilings, each measured

**Disk: 3.0 GB/s and it does not go faster.** `direct=True` uses FILE_FLAG_NO_BUFFERING,
bypassing the Windows file cache. Turning it off measures 1.06 GB/s (decode 726 -> 1103
ms), so it stays on. Thread count is saturated at 8: 8/16/32 readers give 2.88/2.99/2.89
GB/s. A naive test that reads the same files twice measures 5.03 GB/s -- that is the OS
page cache, not the disk.

**Cache hit rate: 52.5% is the ceiling.** Simulated over 12 decode steps (2880 expert
visits) with a perfect LRU:

    cap=  400 ( 7.3 GB): 31.7%
    cap=  862 (15.8 GB): 49.3%    <- the current plan
    cap= 1500 (27.5 GB): 52.5%
    cap= 6000 (110  GB): 52.5%    <- saturated

Going from 15.8 GB to 27.5 GB buys 3.2 points, i.e. 2.49 -> 2.33 GB of disk traffic, i.e.
about 6% of a token, for 11.7 GB more RAM on a 47.2 GB box. The ceiling comes from the
model's routing, not the implementation: 15360 experts exist, a token touches 240 of them,
and the chance the next token touches the same ones is what it is.

**The load shape matters and this table is the pessimistic one.** Feeding twelve
consecutive token ids (500, 501, ...) instead of tokens the model chose for itself moves
the same measurement from 49% to 74%, because adjacent ids produce similar hidden states
and therefore correlated routing. Real decoding looks like the model choosing, so 49-52%
is the number to quote; 74% is an artefact of a synthetic input. For what it is worth, on
the correlated load ARC measures 72.9% against a perfect-LRU 65.3% at equal capacity, so
the adaptive policy is doing better than the textbook one.

**Launch overhead: 6.1 ms, already at the driver floor.** 370 launches per token at
16.5 us each is cuLaunchKernel itself -- measured by timing submissions without
synchronising. Even reducing launch count to zero (impossible) would only buy 18%.

## Corrections to earlier claims in this repository

These were all wrong at some point today and are recorded so they are not repeated:

1. **"H2D is 808 ms of the step."** Profiler attribution, not transfer time. An isolated
   pinned->device copy costs 3-4 us of fixed cost plus ~28 GB/s; the 1.2 s attributed to
   `Memcpy HtoD` is queued time on the copy stream while the disk read is in flight.
   PCIe traffic for experts is zero -- they are consumed from pinned host bytes.

2. **"The expert cache holds 42.9%."** Simulated over 6 steps, too short. Twelve steps
   gives 52.5%. Single-step measurements are worse still: they report 100% because the
   cache has not finished filling (`ram entries 739 -> 913` over eight steps).

3. **"Parallel reads are running at 2.73 GB/s, 84% below the 5.03 ceiling."** Two errors:
   the 2.73 figure was a cumulative average that included prefill, and the 5.03 was
   page-cache. Per-step decode is 4.18 GB/s against a cold-disk 3.0, which is consistent.

4. **"The tree's `es._pool` is None, so the host pool was never built."** `pool` belongs
   to `ExpertCache` (`es.ram._pool`, a 15.05 GB BlobPool). `getattr(x, '_pool', None)`
   returns None for an attribute that does not exist, which is not evidence of anything.
   Confirm a mechanism by counting its calls, not by inspecting what it might depend on.

5. **"Launch marshalling costs 19 us per call."** It costs about 2.5 us: 16.5 of the
   19 is the driver call, which no amount of Python work removes.

## What was actually improved today

All of it is bit-exact against the previous path (verified per call site):

    act_quant        26.4 -> 18.3 us      BoundLaunch
    moe_act6         27.5 -> 11.2 us      BoundLaunch
    moe_act_q6       34.6 -> 13.2 us      BoundLaunch
    sinkhorn_hc     103.0 -> 93.3 us      BoundLaunch
    fp8_gemv_bs      248 calls/step       BoundLaunch
    hc_post          80 calls/step        BoundLaunch
    GEMV kernel      32.1 -> 70.9 GB/s    per-lane block ownership + 128-bit loads

    slow-path launches per decode step: 980 -> ~370
    end to end: ~808 -> ~790 ms (inside the 782-850 run-to-run spread)

The bound-launch work is worth about 6 ms/token. The GEMV work is worth almost nothing
here and everything on a card that holds the model: at 24 GB the decode working set lives
in the VRAM slot pool, so kernel time is the whole token.

## The honest bottom line

On this machine decode is disk-bound and the disk is at its ceiling. The two levers left
are both hardware:

    MTP speculative decoding   -61% expert bytes at 50% acceptance   (+72%)
                               needs 7.6 GB of draft stages; 8 GB cannot hold them
    24 GB or more              zero PCIe, zero disk for the expert table

Neither is a software problem. Everything above was measured trying to find one anyway.
