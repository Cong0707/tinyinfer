# Why this project is shaped the way it is

## The model author is the authority

Every semantic decision resolves toward DeepSeek: `config.json` for parameters, DeepInfer
(their serving stack) for operator behaviour, the released reference kernel for numerics.
Where another implementation disagrees (e.g. the plain HF rotary table omitting YaRN's
mscale), the vendor's stack wins and the difference is documented rather than smoothed
over. NVIDIA's work is read as an execution-structure reference (how to stream, how to
overlap, which shape a kernel should have) and re-expressed in TileLang, because the
goal is one source that runs on many accelerators, not one vendor's binary.

## Levels instead of a switch

Optimisation is a ladder, not a flag: level 0 is a reference implementation, level 1
ports mechanisms that are already proven upstream, level 2 is this engine's own features,
level 3 is approximation. A machine that cannot hold every layer needs level 2 to start
at all (streaming and expert caches are level-2 features), and that is stated in the
refusal message rather than silently degrading.

## Zero configuration

The planner reads VRAM, RAM and disk and sizes residency, ring, cache and chunking. If a
zero-config run misbehaves, the fix belongs in the planner or the adapter, not in a hand-
tuned flag. Every environment variable in docs/ENV.md exists for measurement or for
promoting a mechanism after it has been validated; none is required.

## Measure, then claim

Every number in docs/ comes from a run on this machine with the disk rate measured in the
same run, because a fetch-bound engine's tok/s is a property of the disk as much as of the
code. The IO arithmetic (docs/lower_bounds.md) is the invariant every speed claim is
checked against, and the measured non-goals in docs/ROADMAP.md exist so the same dead ends
are not re-explored.

## Correctness is not negotiable

A kernel whose reduction order differs by 4e-3 is opt-in, even when it is faster; a
multi-row kernel that is exact on synthetic weights but 35% off on real ones is disabled
by default even though it is 1.6x faster. The single-row paths that decode uses are
exact, and the parity table (docs/parity.md) records what that means per component.
