# Documentation index

| file | contents |
| --- | --- |
| STATUS.md | what runs, measured numbers, open items (read this first) |
| lower_bounds.md | the hardware IO floor computed from measured constants |
| analysis.md | routing predictability, speculative-decoding ceiling, top-k floor |
| decode_speed.md | every optimisation and what it changed, with before/after numbers |
| benchmarks.md | benchmark history and rules of thumb (disk rate is part of every row) |
| parity.md | numeric parity per component against DeepInfer |
| kernels_provenance.md | every kernel: official port / re-implementation / why |
| level2_mechanisms.md | the level 1/2 mechanisms and their upstream references |
| flare_analysis.md | NVIDIA Flare (b12x) mechanisms: ported, rejected, why |
| official_repos.md | what each official DeepSeek repository offers |
| architecture.md | module map, streaming model, expert path, sessions |
| ENV.md | every environment switch (measurement and A/B only) |
| changelog.md | chronological notes of each round |
| CONTRIBUTING.md | ground rules, workflow, how to add an optimisation |

## Reading order

1. `STATUS.md` — current capability and known gaps.
2. `lower_bounds.md` + `analysis.md` — why the speed is what it is.
3. `parity.md` — why the numbers are trusted.
4. `architecture.md` — how the engine is put together.

## Reproduce

    PYTHONPATH=src python -m tinyinfer.bench --model <dir>
    PYTHONPATH=src python examples/generate.py --model <dir>
    PYTHONPATH=src python -m pytest tests -q