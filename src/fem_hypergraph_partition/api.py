"""Stable release boundary; backend selection is explicit."""
import importlib
from .model import Hypergraph, capacity, score


def partition(graph, *, epsilon=.03, seed=21, steps=12000, cold=144,
              replicas=48, backend='reference', device='cpu', capture=True):
    if not isinstance(graph, Hypergraph):
        raise TypeError("graph must be a Hypergraph")
    capacity(graph.n, epsilon)
    for name, value, minimum in [('seed',seed,0), ('steps',steps,4), ('cold',cold,0), ('replicas',replicas,1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if backend == 'reference':
        if device != 'cpu':
            raise ValueError("the portable reference uses CPU; request backend='native' for CUDA")
        from .reference import partition_reference
        return partition_reference(graph, epsilon, seed, steps, cold, replicas)
    if backend != 'native':
        raise ValueError("backend must be 'reference' or 'native'")
    if replicas != 48:
        raise ValueError("this native prototype uses the fixed 48-replica portfolio")
    try:
        engine = importlib.import_module('fem_hypergraph_partition._engine.entry')
    except ModuleNotFoundError as exc:
        if not exc.name.startswith('fem_hypergraph_partition._engine'):
            raise
        raise RuntimeError("Install a matching compiled backend wheel; the reference backend remains available.") from exc
    result = engine.partition(graph.n, graph.edges, epsilon, seed, steps, cold, device, capture)
    # Always recompute the advertised cut and imbalance outside the engine.
    measured = score(graph, result['partition'], epsilon)
    for key in ['cut', 'block_sizes', 'feasible']:
        if result[key] != measured[key]:
            raise RuntimeError("compiled backend output failed independent scoring")
    result.update(measured)
    return result
