# Reproduction scope

## Complete catalytic solver

Version 0.1.0a4 makes the protected compiled solver the default. It retains
the exact cut-net QUBO, all original vertex and net variables, correlated
auxiliary/slack blocks and the full 48-replica portfolio. The three groups
use local, unscaled-mode and normalized-mode directions with matched random
starts. Pair, auxiliary, collective and local-temperature schedules are
retained and vanish before final readout. Arithmetic is explicitly FP64.

The standard budget is 12,000 annealing updates and 144 cold updates, with
24 implicit-root iterations and checkpoints every 128 updates plus phase
boundaries. `partition` reports the best raw feasible FEM endpoint. Separate
`selected_partition`, `selected_cut` and `selected_feasible` fields retain
the result of original-polynomial descent on all replicas.

Install a matching compiled wheel and its runtime dependencies:

```bash
python -m pip install 'fem-hypergraph-partition[native] @ file:///absolute/path/to/matching.whl'
fem-hypergraph examples/tiny.hgr --device cuda --epsilon .03
```

The CPython 3.10 SM80 wheel includes fused CUDA kernels. The CPython 3.11
wheel runs the same catalytic policy through Torch. Neither silently falls
back to the catalyst-off baseline. An absent or outdated compiled engine
raises an error. Binary identities and build/check scope are recorded in
[native_artifact.json](benchmarks/native_artifact.json).

## Modes and numerical exceptions

Fresh mode construction uses the original incidence operators. To retain
an existing basis and its eigenvalues, use
`--modes UNSCALED.npz NORMALIZED.npz`. Each cache needs `modes`, `inactive`
and `eigenvalues` arrays, or the recorded eigenvalues in an adjacent
`modes.json`. The engine checks support, balance, orthogonality and residuals
against the supplied hypergraph before using a cache. Modes supply temporary
couplings and do not initialize a binary partition.

The manuscript's scircuit preprocessing exception requires its certified
mode cache. Its imbalance-band root exception is explicitly selectable with
`--root-iterations 64 --central-root`. It is not applied to every input.
Historical mode-construction charges and failed-attempt charges in the
paper remain distinct from the elapsed time of loading cached arrays.

## Baselines and checks

The NumPy catalyst-off FEM remains available with `--backend reference`.
It implements four-state net Gibbs blocks, correlated slack with binary
multiplicities, count variance, implicit updates and per-replica readout.
The older independent-Bernoulli Adam example uses `--backend educational`.
Both are explicit controls, not substitutes for the catalytic solver.

```bash
python -m pip install '.[test,plots]'
python -m pytest -q tests
python docs/plot_benchmarks.py
python docs/check_benchmarks.py --data-dir /path/to/hypergraphs
```

Tests cover small exact QUBOs, independent cut/capacity scoring, base FEM
mechanics and default-backend dispatch. When a compiled wheel is installed,
they also execute its full portfolio on small inputs. These checks do not
replace a full performance experiment. The author requested skipping a new
large public/private comparison for this release, so no new 30-input native
reproduction or quality improvement is claimed.

## Published measurements

[paper_tables.json](benchmarks/paper_tables.json) retains the frozen
30-input QUBO study at 0%, 1% and 3% allowed imbalance, with native cuts,
actual imbalance, input hashes and charged times. Plots redraw these saved
measurements; they do not rerun the solver. Large inputs, saved mode/state
arrays and reference partitions remain separate research artifacts. Obtain
inputs from their original distribution and verify their hashes.

## Branches and synchronization

The private upstream keeps `qubo-private-dev` separate from the clean
`qubo-public-release` branch. Only the audited clean branch synchronizes to
`FemHyPar/qubo-hypergraph-partition:main`. The workflow checks the release
manifest, protected ancestry and public tests before an ordinary
fast-forward push. It refuses diverged public history and never force-pushes
research branches or tags. Synchronization uses `PUBLIC_RELEASE_SSH_KEY`;
`PUBLIC_RELEASE_TOKEN` is an optional narrowly scoped fallback. Credentials
stay in GitHub Actions secrets.
