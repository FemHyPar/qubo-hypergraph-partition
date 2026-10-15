# FemHyPar

Free-energy machine for balanced hypergraph partitioning.

This repository implements an exact **QUBO** formulation of the cut-net
objective. Inspired by temporary catalyst interactions in quantum annealing,
the research solver changes the intermediate classical free energy and
restores the target QUBO before readout. It starts from random fields on the
original hypergraph; KaHyPar supplies independent references.

## Quick start

```bash
python -m pip install 'fem-hypergraph-partition[native] @ file:///absolute/path/to/matching.whl'
fem-hypergraph examples/tiny.hgr --device cuda --steps 12000 --cold 144 --replicas 48 --epsilon .03
```

```python
from fem_hypergraph_partition import Hypergraph, partition

graph = Hypergraph.read_hgr("examples/tiny.hgr")
result = partition(graph, steps=12000, backend="native", device="cuda")
print(result["cut"], result["observed_imbalance_percent"], result["feasible"])
```

The default backend runs the complete catalytic solver: 48 replicas in
local, unscaled-mode and normalized-mode groups, FP64 arithmetic, the
paper's temperature and penalty schedules, and checkpoints every 128
updates. The primary result is the best raw feasible FEM endpoint; the
separate `selected_*` fields report polynomial descent.

Install a matching protected wheel from
[Releases](https://github.com/FemHyPar/qubo-hypergraph-partition/releases).
Missing or outdated native engines raise an error. The portable catalyst-off
baseline remains available only with `backend="reference"` explicitly.
Inputs support unweighted `.hgr` format 0, two blocks and unit vertex loads.
Certified mode caches can be supplied with `--modes UNSCALED NORMALIZED`;
the documented numerical exception uses `--root-iterations 64 --central-root`.

## Paper and results

Read the working [manuscript](paper/manuscript.pdf) and
[supplement](paper/supplementary.pdf) by Baijian Yao and Xiwei Pan.
The QUBO study uses 30 original hypergraphs, 48 replicas and matching
0%, 1% and 3% capacity settings. It improves 15 recorded KaHyPar cuts at
exact balance, and none at 1% or 3%. All primary results use FP64; later
precision controls are reported separately.

- [Cut scores and measured imbalance](docs/images/qubo_quality.png)
- [Complete charged runtimes](docs/images/qubo_runtime.png)
- [Machine-readable results and input hashes](docs/benchmarks/paper_tables.json)
- [Reproduction scope and native artifact details](docs/reproducibility.md)

The solver is classical. The experiments do not establish quantum advantage
or an overlap-gap bypass. Native HUBO methods are developed in the separate
[HUBO repository](https://github.com/FemHyPar/hubo-hypergraph-partition).

## Tests

```bash
python -m pip install '.[test]'
python -m pytest -q tests
```

## Licence

[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial licensing:
**byao@connect.ust.hk**. Third-party terms are listed in [NOTICE](NOTICE).
The protected engine is distributed as compiled code; public Python files
provide the interface, scoring and reference solver.
