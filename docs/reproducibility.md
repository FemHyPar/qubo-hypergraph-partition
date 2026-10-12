# Reproduction scope

The public reference now implements the manuscript's **catalyst-off base**.
Its four-state net Gibbs blocks, correlated slack distribution with binary
multiplicities, count-variance contribution, manual implicit update, nearby
fields, degree-scaled step sizes, late penalty ramp and per-replica
checkpoint/readout rules replace the earlier Adam example as the default.
The old independent-Bernoulli Adam implementation is named `educational`.
All temporary energy terms and local-temperature offsets are off in the
reference. This is a mechanism baseline, not the catalytic research policy.

Public tests verify the Gibbs energy/entropy identity, slack multiplicities
against binary enumeration, implicit roots against independent bisection,
every polynomial one-bit move at readout, and the two 12-vertex objective
counterexamples (924 balanced assignments each). The private integration
audit separately compares the public CPU base with catalyst-off research
trajectories. Its aggregate certificate and source hashes are in
[base_validation.json](benchmarks/base_validation.json).

The strict polynomial fixes the vertex count to floor(n/2), including odd
inputs, as in the paper. At nonzero allowed imbalance it uses bounded binary
slack. Independent scoring accepts either orientation of a balanced odd
partition. Inputs and the large-instance experiment currently use unit net
and vertex weights; weighted theoretical examples require the private
research implementation. The `.hgr` parser deliberately supports format 0.

```bash
python -m pip install '.[test,plots]'
python -m pytest -q tests
python docs/plot_benchmarks.py
python docs/check_benchmarks.py --data-dir /path/to/hypergraphs
fem-hypergraph examples/tiny.hgr --backend reference --epsilon 0 \
  --steps 12000 --cold 144 --replicas 48 --root-iterations 24
```

The CLI exposes replica count, cold steps and CUDA capture. The native
prototype owns its separately validated numerical safeguard policy; the
CPU root-iteration option cannot override that policy. The reference has
no GPU acceleration. CPU time from this implementation cannot replace the
A100 timing rows in the paper.

## Results and artifacts

[paper_tables.json](benchmarks/paper_tables.json) exports the native cuts,
actual imbalance, measured times and input hashes for the 30-input QUBO
study and separate three-input HUBO controls. The plot script reproduces
the README bars from these aggregates. It does **not** rerun the solver.
The KaHyPar times are measured process times, with per-instance soft budgets
and differing hardware; they are not assumed runtimes copied from budgets.

Large input files, reference partitions, full replica states, coherent
simulation records and dataflow trials remain in the private research
archive. Obtain benchmark inputs from their original distribution and
check the exact hashes before comparing; do not substitute graph edge lists
for hypergraph `.hgr` files. Peer-review reproduction requires the validated
`private-release` snapshot and its artifact archive.

The CPython 3.10/Linux x86-64 SM80 artifact retains the validated fused
0.1.0a2 engine; version 0.1.0a3 updates the public API and CPU baseline.
[native_artifact.json](benchmarks/native_artifact.json) records binary identity
and validation scope. Its protected implementation is distributed only as
compiled code. Seven paired large-instance configurations have been audited;
the full 30-input performance protocol, scircuit preprocessing exception and
new direct HUBO solver are **not** validated public-native reproductions yet.
The runtime needs compatible Torch/SciPy; the private CuPy mode-construction
experiments are not exposed by this prototype. No public reference runtime,
aggregate plot or small test establishes full benchmark reproduction.

## Branches and synchronization

The private upstream keeps `private-dev`, `private-release` and
`public-release`. This clean branch has no research-source ancestors. Only
it can synchronize into `yao-baijian/qubo-hypergraph-partition:main`.
The private repository needs the `PUBLIC_RELEASE_TOKEN` Actions secret,
restricted to Contents write on that public repository. The workflow checks
the exact SDK manifest, protected ancestry and public tests before an ordinary
fast-forward push. It never pushes private branches, tags or force updates.

This is synchronization, not a GitHub fork: private repository forks remain
private according to [GitHub's fork rules](https://docs.github.com/en/pull-requests/reference/forks).
Store credentials in [GitHub Actions secrets](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token),
never in tracked files or messages.
