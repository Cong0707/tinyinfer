# Release checklist

Run before calling a build usable; each line has a command or a doc to read.

## Code

- [ ] `powershell -File scripts/check.ps1` passes (syntax + fast tests).
- [ ] No new CUDA/CuTe/Triton/vendor-binary dependency (docs/kernels_provenance.md).
- [ ] New kernels are behind a tier flag or env switch unless bit-exact.
- [ ] No machine-bound constant added; the planner reads the hardware.

## Numerics

- [ ] `probe_parity1.py` window-layer attention = 0.000000.
- [ ] `probe_expert_fp4.py` expert FFN = 0.000000.
- [ ] `probe_gate2.py` gate weights and indices equal.
- [ ] `probe_hc.py` hyper-connection ops = 0.000000.
- [ ] `probe_rope_di2.py` RoPE vs DeepInfer < 1e-5.
- [ ] Compressed layers within the documented 0.87% / 1.42%.

## Performance

- [ ] `python -m tinyinfer.bench --model <dir>` completes; record disk rate + tok/s.
- [ ] tok/s within ~25% of the IO floor for the measured disk rate (docs/lower_bounds.md).
- [ ] No new regression vs docs/benchmarks.md at a comparable disk rate.

## Service

- [ ] `python -m tinyinfer.cli serve` answers `/health` and `/v1/completions`.
- [ ] 503 while loading, 200 after.
- [ ] One request at a time is documented (single KV session).

## Docs

- [ ] docs/STATUS.md numbers match the last bench run.
- [ ] docs/changelog.md has an entry for the change.
- [ ] docs/ROADMAP.md updated if a trade-off or open item changed.
- [ ] README quick-start still reproduces.

## Known-acceptable state on 8GB/47GB

- decode ~0.4-0.6 tok/s depending on disk; 1.66s was at 4.5GB/s.
- long prompts at chunk=1 take minutes (30GB of experts).
- base-model output style; question-shaped prompts work.
- the open items in docs/STATUS.md (batch-vs-row, session drift, MTP).
