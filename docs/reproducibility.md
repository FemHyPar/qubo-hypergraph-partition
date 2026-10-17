# Reproduction scope

## Current FP32 paper protocol

Version 0.1.0a5 includes the paper's single-trajectory FP32 solver and
explicit conditional-QUBO CPU refinement. It preserves all original
vertices and net auxiliaries; strict balance uses the complement-symmetric
penalty and relaxed capacities use bounded binary slack. Temporary
interactions vanish before terminal original-QUBO verification.

Install a matching compiled wheel from the project's Releases page and
its runtime dependencies:

```bash
python -m pip install 'fem-hypergraph-partition[native] @ file:///absolute/path/to/matching.whl'
fem-hypergraph examples/tiny.hgr --device cpu --epsilon .03
```

The CLI defaults to the FP32 protocol, with one trajectory, a maximum of
3,000 annealing plus 144 terminal updates, 64 safeguarded count-root
iterations and observable early stopping. The Python entry is:

```python
from fem_hypergraph_partition import partition_fp32
result = partition_fp32("examples/tiny.hgr", epsilon=.03, seed=21)
print(result["cut"], result["observed_imbalance_percent"])
```

For the 30 paper inputs, a path is matched by SHA-256 to the frozen
instance/capacity policy selected before evaluation. Other inputs use a
documented generic profile; passing an in-memory Hypergraph also uses
that generic profile unless replica and arm are specified. The native
input reader, sparse updates, selected strategy, stopping and refinement
are compiled from the private implementation. A missing or outdated
engine raises an error and never substitutes the catalyst-off baseline.

The CPython 3.11 Linux wheel contains the protected modules, native
input/refinement library and SM80 CUDA library. CPU execution uses Torch;
CUDA execution requires a matching architecture. Validation scope and
binary hashes are recorded in [native_artifact.json](benchmarks/native_artifact.json).
A source package does not contain the protected engine. Support for an
additional Python ABI or GPU architecture requires a matching wheel.

## Directions, calibration and clocks

Fresh mode construction uses the original sparse incidence operators.
For reproducible cached directions, pass
`--modes UNSCALED.npz NORMALIZED.npz`, with the adjacent JSON provenance
files containing each mode-file hash, input hash and eigenvalues. The
FP32 entry checks file and input identities. The paper's retained bases
also have independent support, orthogonality and residual audits.
Modes define temporary couplings and never supply a binary initial
partition. The large input and mode arrays are separate research artifacts.

The paper's process clock includes interpreter startup, input, native
model setup, mode loading, graph capture, search, checkpoints and CPU
refinement. Prior mode construction and instance-specific strategy
calibration are excluded and disclosed in Supplementary Note 4. This is
a cached-direction solve, not a complete first-use cost. CPU refinement
repairs infeasibility when necessary and retains the best feasible prefix
of gain-based moves; its cost is charged to every measured solve.
Independent audits, archival and the random-label refinement control are
outside the solver clock.

## Published measurements

[paper_tables.json](benchmarks/paper_tables.json) provides the current
30-input, three-capacity, three-trial comparison. Table 2 reports each
solver's best native cut of three trials and the imbalance of that
partition. Table 3 lists the original vertex count for each input and the
median individual process wall time from those same trials. That median is not time to the best-of-three cut;
three serial trials cost their summed times.

The [complete FP32 measurements](benchmarks/fp32_primary_2026-10-17.json)
retain all 270 FEM trials, all 270 single-call KaHyPar-K trials,
checkpoint traces, stopping outcomes, raw/refined cuts, random-label
refinement controls and the independent 12,681-state audit receipt.
All refined FEM outputs are feasible; eleven strict trials require
explicit repair after failing to produce a feasible raw checkpoint.
Neither those trials nor unsuccessful target searches are omitted.
Each capacity has a separate equal-weight mean over all 30 inputs.

Reference-quality targets use the best of the three KaHyPar-K calls at
the same capacity. They are empirical cuts, not certified optima. A 99%
quality target is distinct from a 99% success probability, and three
trials cannot establish high-reliability time-to-solution claims.
The direct strict-balance KaHyPar configuration is a limited baseline;
stronger slack-then-repair pipelines can return substantially better
exactly balanced partitions. Different GPU/CPU hardware and excluded
preprocessing also limit speedup interpretations.

The [dated table archive](benchmarks/paper_tables_before_fp32_primary_2026-10-17.json)
retains the historical FP64 portfolio and separate symmetry follow-up.
[Earlier mechanism controls](benchmarks/historical_controls_2026-10-16.json)
and the [matched stopping diagnostic](benchmarks/time_quality_2026-10-16.json)
retain their own precision, seeds and budgets. They are not pooled into
the current tables. Exploratory dynamic-penalty variants are not enabled
by the paper protocol.

## Interfaces and verification

The historical 48-replica FP64 interface remains explicitly available as
`partition(...)` or `--protocol legacy`. The portable catalyst-off FEM
requires `--backend reference`; the older Adam example requires
`--backend educational`. These controls do not reproduce the current
catalytic protocol and are never silent fallbacks.

```bash
python -m pip install '.[test,plots]'
python -m pytest -q tests
python docs/plot_benchmarks.py
python docs/check_benchmarks.py --data-dir /path/to/hypergraphs
```

Tests verify exact QUBO costs, capacity checks and dispatch. Installed
compiled-wheel checks additionally exercise the FP32 path and historical
portfolio on small inputs, with independent scoring. Their scope is
reported separately from the complete private A100 benchmark; a new
large public/private benchmark was not rerun for this release.

## Branches and synchronization

Only the audited clean `qubo-public-release` branch synchronizes to
`FemHyPar/qubo-hypergraph-partition:main`. `qubo-private-dev` keeps the
private implementation and research history. The workflow verifies the
manifest, protected ancestry and public tests before an ordinary
fast-forward push. It refuses diverged public history and never
force-pushes public changes. Compiled wheels are separate release assets;
source synchronization does not upload those assets automatically.
