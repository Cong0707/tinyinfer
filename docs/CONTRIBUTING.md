# Contributing / working on tinyinfer

## Ground rules this project keeps

1. **TileLang + torch only.** No CUDA C++, no CuTe DSL, no Triton, no FlashInfer /
   DeepGEMM / b12x / flash-mla binaries. NVIDIA implementations are read for the
   algorithm and re-expressed in TileLang (docs/kernels_provenance.md).
2. **The model author is the authority.** Semantics come from DeepSeek's `config.json`
   and DeepInfer; NVIDIA/vLLM are execution-structure references only.
3. **Levels 0-2 must stay bit-comparable to the reference.** A kernel whose reduction
   order differs (e.g. the TileLang norm leg, ~4e-3) is opt-in, never a default.
4. **No machine-bound configuration.** The planner reads VRAM/RAM/disk; flags exist for
   measurement and A/B only (docs/ENV.md).
5. **Measure before claiming.** Every number in docs/ comes from a run on this machine
   with the disk rate measured in the same run.

## Workflow

    powershell -File scripts/check.ps1                      # syntax + fast tests
    powershell -File scripts/check.ps1 -Benchmark -Model <dir>
    PYTHONPATH=src python -m tinyinfer.bench --model <dir>  # JSON metrics

- `tests/` holds the fast suite; checkpoint-backed tests are marked and can be excluded
  with `-k 'not streaming_session'`.
- Probes that need the real model live in the repo root as `probe_*.py`; they are the
  evidence for docs/parity.md and docs/analysis.md.

## Where to look first

| file | why |
| --- | --- |
| docs/STATUS.md | current capability and open items |
| docs/ROADMAP.md | ordered next steps and measured non-goals |
| docs/architecture.md | module map and data flow |
| docs/lower_bounds.md | the IO arithmetic every speed claim is checked against |
| docs/ENV.md | every switch |

## Adding an optimisation

1. Write the probe first (`probe_*.py`) and record the reference number.
2. Put it behind a tier flag or an env switch, defaulting to the aligned behaviour.
3. Verify parity (0.000000 or an explained residual) and measure the effect.
4. Document it in docs/changelog.md and, if it changes a trade-off, docs/ROADMAP.md.
