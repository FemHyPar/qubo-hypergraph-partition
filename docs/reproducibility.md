# Reproduction scope

## Current FP32 paper protocol

The current paper tables use the standalone C++/CUDA implementation,
with one device-controlled graph per trajectory. Its fresh-process
timings are distinct from the CPython wheel interface described below;
the wheel's earlier Python frontend must not be assigned these timings.

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

The paper's process clock starts before launching the native executable
and includes input/model setup, cached-mode loading, CUDA initialization,
graph preparation, search, host readout and CPU refinement. The runtime
breakdown records graph execution through completed host readout as one
component; that interval alone is not full solve latency.
Prior mode construction and instance-specific strategy
calibration are excluded and disclosed in Supplementary Note 4. This is
a cached-direction solve, not a complete first-use cost. CPU refinement
repairs infeasibility when necessary and retains the best feasible prefix
of gain-based moves; its cost is charged to every measured solve.
Independent audits, archival and the random-label refinement control are
outside the solver clock.

The current standalone executable checks every 64 updates and enters
terminal verification after three consecutive equal feasible current
cuts, each equal to the best feasible cut. Its CPU refiner has fixed
pass/move limits and charges all actual time. These new standalone
measurements do not change the older wheel frontend or its archived
validation; a wheel must be rebuilt and verified before receiving the
new protocol or being assigned these timings.

## Published measurements

[paper_tables.json](benchmarks/paper_tables.json) provides the current
30-input, three-capacity, three-trial comparison. Table 2 reports each
solver's best native cut of three trials and the imbalance of that
partition. Table 3 lists the original vertex count for each input and the
median individual full process time from those same trials. Those medians are not time to the best-of-three cut;
three serial trials cost their summed times.

The [complete FP32 measurements](benchmarks/fp32_primary_2026-10-17.json)
retain all 270 FEM trials, all 270 single-call KaHyPar-K trials,
returned endpoints, stopping outcomes, raw/refined cuts, the matched
random-label refinement controls and the independent 11,813-state audit
summary. All refined FEM outputs are feasible; fourteen strict trials require
explicit repair after failing to produce a feasible raw checkpoint.
Neither those trials nor unsuccessful target searches are omitted.
Each capacity has a separate equal-weight mean over all 30 inputs.
Only the returned refined endpoint establishes host-available target
arrival. Internal device checkpoint times are not earlier output times.

Reference-quality targets use the best of the three KaHyPar-K calls at
the same capacity. They are empirical cuts, not certified optima. A 99%
quality target is distinct from a 99% success probability, and three
trials cannot establish high-reliability time-to-solution claims.
The direct strict-balance KaHyPar configuration is a limited baseline;
stronger slack-then-repair pipelines can return substantially better
exactly balanced partitions. Different GPU/CPU hardware and excluded
preprocessing also limit speedup interpretations.

The [Python-reference table archive](benchmarks/paper_tables_before_native_cuda_2026-10-17.json)
preserves the earlier frontend's cut and timing cohort. The
[dated table archive](benchmarks/paper_tables_before_fp32_primary_2026-10-17.json)
retains the historical FP64 portfolio and separate symmetry follow-up.
[Earlier mechanism controls](benchmarks/historical_controls_2026-10-16.json)
and the [matched stopping diagnostic](benchmarks/time_quality_2026-10-16.json)
retain their own precision, seeds and budgets. They are not pooled into
the current tables. Exploratory dynamic-penalty variants are not enabled
by the paper protocol.

The [runtime breakdown](images/runtime_breakdown.png) orders all 30 inputs
by original vertex count. Every stacked bar uses the actual median-time
trial, so its measured stages sum to the corresponding Table 3 total.
A separate [resident-process diagnostic](benchmarks/warm_process_fp32_2026-10-17.json)
covers 27 requests on IBM01, IBM10 and scircuit, plus one cold primer.
Reusing dependencies and the CUDA context saves a median 1.97 seconds per
request while rebuilding each input/model and update graphs and reloading
cached modes. All binary checkpoints match the fresh-process runs.
The primary tables use fresh native processes, rather than those
Python resident-request times. An independent-input concurrency pilot
shares one CUDA context across eight jobs. Four streams reduce GPU
queue time from 14.18 to 6.29 seconds (2.25x throughput); including
serial CPU refinement gives 14.52 to 6.62 seconds (2.19x). Median active
search latency increases from 1.73 to 3.02 seconds. These cached queue
measurements are not individual solve latency and have no matched
parallel KaHyPar control yet.

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
