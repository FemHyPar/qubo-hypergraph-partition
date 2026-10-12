# qubo-hypergraph-partition

Hypergraph partitioning divides vertices into balanced blocks while keeping
shared dependencies inside a block. In a circuit netlist, one net can connect
many cells; in a simulation workload, one dependency can connect several
tasks. A cut-net objective charges a net once when it spans blocks, regardless
of how many pins cross. Preserving this many-body structure matters when
translating the problem into a binary energy for quantum optimization.

This project studies whether temporary **catalyst interactions** can improve
search on faithful hypergraph energies. The catalyst changes the intermediate
optimization path and vanishes before the final original-objective phase. A
Free-Energy Machine (FEM) evolves variational probability distributions to
study these paths at practical sizes on GPUs. The current implementation is
quantum-inspired classical computation; small coherent simulations in the
research study examine related Ising Hamiltonians separately.

The original motivation includes **HUBO catalysis**. The current large-instance
benchmark and released solver implement **exact QUBO catalysis**, using
auxiliary bits to encode the same cut-net objective. The private research implementation also evolves the native
HUBO without net auxiliaries; three matched seen-instance comparisons and
a fixed replica-timing mixture are included in the results below. The solver starts from the original incidence
structure and random fields. KaHyPar supplies independently measured
references and does not initialize the search.

## HUBO, QUBO and FEM

The natural higher-order unconstrained binary optimization (**HUBO**) energy
for cut nets is
`sum_e w_e [1 - prod_(i in e) x_i - prod_(i in e) (1-x_i)]`.
It is zero for an uncut net and adds its weight for a cut net. Large nets
therefore introduce high-order products directly.

The benchmark solver replaces each net term with an exact quadratic
unconstrained binary optimization (**QUBO**) encoding. For a net with size
`r` and vertex count `k`, two binary auxiliaries give
`min[u,v] {1 + (r-1-k)u + (k-1)v} = 1[0<k<r]`.
Minimizing the auxiliaries recovers the original cut for every binary vertex
assignment. A quadratic count penalty, with bounded slack bits for allowed
imbalance, enforces capacity at global minima. The heuristic may return an
infeasible candidate; outputs report the actual cut, block sizes, imbalance
and feasibility.

Exact binary-objective equivalence does **not** imply identical variational
free energies or search dynamics. Auxiliary variables, their entropy and
the allowed correlations change the intermediate FEM landscape. Our current
catalysts are quadratic and vanish at the endpoint. FEM is the solver for
these energies, rather than another objective encoding.

![Objective construction and exact cut-net encoding](docs/images/objectives.png)

## Relation to KaHyPar and quantum optimization

[KaHyPar](https://kahypar.org/) uses multilevel contraction, initial
partitioning and refinement. Our search retains the original variables and
evolves probability distributions under a temporary catalytic energy.
These are different optimization dynamics for the same final cut and
balance constraints. QUBO encoding alone establishes no advantage over
KaHyPar. We compare feasible cuts, measured imbalance and charged solver
time at matching capacity settings; the present catalyst policy does not
consistently outperform the recorded KaHyPar references.

Here, "catalyst" means a temporary energy term. The implementation does not
yet enforce restoration of a separate catalyst-replica distribution to its
initial state. That proposed form of catalysis requires its own model and
validation. Keeping a HUBO objective also does not by itself make a solver
quantum: quantum dynamics require an explicitly specified quantum driver
and quantum evolution, beyond the classical GPU FEM search.

An overlap-gap property (OGP) concerns the possible overlaps between
low-energy configurations. It is not simply a flat gradient surface, and
its existence or an algorithmic obstruction must be established for the
particular problem ensemble and algorithm class. The catalyst is a proposed
way to alter search paths; these experiments do not establish an OGP bypass
or quantum advantage.

![Catalyst mechanism and overlap geometry](docs/images/method.png)

The overlap and landscape panels illustrate separated solution families and
temporary path deformation. They are schematic, rather than measurements
of a benchmark's energy surface.

## Install and run

```bash
python -m pip install .
fem-hypergraph examples/tiny.hgr --steps 1000 --cold 144 --replicas 48 --epsilon .03
```

```python
from fem_hypergraph_partition import Hypergraph, partition

graph = Hypergraph.read_hgr("examples/tiny.hgr")
result = partition(graph, steps=1000, backend="reference")
print(result["cut"], result["observed_imbalance_percent"], result["feasible"])
```

The default `reference` backend implements the manuscript’s catalyst-off
base FEM in NumPy on CPU: exact four-state net Gibbs blocks, a correlated
slack block, the implicit count update, nearby starts, the late penalty ramp,
128-update and phase-boundary checkpoints per replica, and original-QUBO
one-bit descent for every replica. Catalyst interactions and local-temperature
offsets are all off. This portable implementation verifies the base equations;
it is not the accelerated solver behind the best results.
The previous independent-Bernoulli Adam example remains available as
`backend="educational"`, and is not a manuscript catalyst-off baseline. Inputs currently
support unweighted `.hgr` format 0, two blocks and unweighted vertices.

## Compiled engine

Install a supplied native wheel matching your Linux architecture and CPython
version, then install its Torch/SciPy runtime dependencies:

```bash
python -m pip install 'fem-hypergraph-partition[native] @ file:///absolute/path/to/matching.whl'
fem-hypergraph examples/tiny.hgr --backend native --device cuda --steps 12000 --cold 144 --replicas 48 --capture
```

The optional backend retains the catalyst portfolio, sparse incidence
operations, manual updates and CUDA Graph replay inside compiled extension
modules. It uses 48 replicas. The SM80 preview compiles fused FP64 implicit
updates, auxiliary responses and sparse incidence products ahead of time; it
uses the same CUDA Graph replay and avoids runtime kernel compilation. The
CPU preview uses Torch operations. The imbalance band uses a conservative
numerical safeguard. Version 0.1.0a3 updates the public CPU/API layer; its supplied SM80 artifact
retains the separately validated 0.1.0a2 compiled engine.
Validation compares original-graph random fields, cut,
measured imbalance and timing with the development implementation. The full
30-instance manuscript protocol remains a separate research benchmark.
A small graph must contain
enough cut-active vertices to construct four modes; use the reference backend
otherwise. An unavailable native backend raises an explicit error.

Public Python files implement I/O, scoring and the reference solver. The
compiled engine does not ship its Python/Cython/C++/CUDA source. Compilation
raises the cost of recovering implementation details; it does not guarantee
secrecy or make reverse engineering impossible.

## Validation

```bash
python -m pip install '.[test]'
python -m pytest -q tests
```

These tests enumerate small QUBOs, check auxiliary minimization and capacity,
and independently recompute returned cut and imbalance. The research protocol
also compares cuts and whole-pipeline times with KaHyPar at matching imbalance
settings. Faster primitive kernels alone establish neither a better optimizer
nor a way around an overlap gap.

## Measured research results

Read the current [manuscript](paper/manuscript.pdf) and
[Supplementary Information](paper/supplementary.pdf) by Baijian Yao,
Xiwei Pan and Wei Zhang. These are working manuscripts, not accepted papers.

These four charts summarize the **research FEM**, not runs of the public
CPU reference. QUBO uses a frozen 48-replica policy on 30 original inputs at
0%, 1% and 3% allowed imbalance. The separate HUBO study uses three **seen**
inputs, one seed, 48 replicas per arm and exact balance. The mixed catalyst
redistributes the same 48 replicas across four fixed timing paths.

![QUBO cut and observed imbalance](docs/images/qubo_quality.png)

QUBO quality: (a) 0%, (b) 1%, (c) 3% allowed imbalance. Bars normalize cut to
the matching KaHyPar reference; lower is better. Every bar annotation gives
**native cut; measured imbalance (%)**. At 0% the policy improves 15 of 30
recorded references; at 1% and 3% it improves none and ties four.

![QUBO charged runtime](docs/images/qubo_runtime.png)

QUBO timing: the same three capacity settings; annotations are seconds.
The horizontal scale is logarithmic to retain slow preprocessing cases.
FEM times charge mode construction, setup, search and all-replica descent.

![HUBO cut and observed imbalance](docs/images/hubo_quality.png)

HUBO quality: (a) *finan512*, (b) *IBM13*, (c) *rajat26*. Annotations give
native cut and measured imbalance; every partition is exactly balanced.
The mixed policy reaches 146, 2,232 and 116, respectively. It still loses to
KaHyPar on the latter two inputs, and these tests do not establish transfer.

![HUBO charged runtime](docs/images/hubo_runtime.png)

HUBO timing: the same three inputs. Off arms do not charge unused mode
construction. FEM runs on A100; KaHyPar uses one i7-12700 CPU thread with
separate soft budgets. These times are not equal-hardware speedups.

The [machine-readable tables](docs/benchmarks/paper_tables.json) retain
native cuts, actual imbalance, input hashes and time for every entry.
[Reproduction scope](docs/reproducibility.md) distinguishes verified base
mechanisms, aggregate plots and the protected performance protocol.
Redraw the charts with `python docs/plot_benchmarks.py` after installing
`pip install '.[plots]'`. No quantum advantage or OGP escape is established.

## Licence

This software is available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a
separate licence. For commercial licensing, contact **byao@connect.ust.hk**.
The terms of `LICENSE` control what constitutes a permitted purpose.
This is a source-available noncommercial release, not an OSI open-source
licence. Third-party software retains its own terms; see [NOTICE](NOTICE).

Normal graph Ising partitioning is maintained separately on the
`hybird-ising-partition` branch; its graph datasets are not hypergraph inputs.

The private upstream has `private-dev` (ongoing research), `private-release`
(validated research snapshot) and `public-release` (this clean SDK). Only
`public-release` is synchronized into this public repository’s `main`.
