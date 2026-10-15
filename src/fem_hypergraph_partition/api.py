"""Stable release boundary; backend selection is explicit."""
import importlib
from .model import Hypergraph, capacity, score


def partition(graph, *, epsilon=.03, seed=21, steps=12000, cold=144,
              replicas=48, backend='native', device='cpu', capture=True, root_iterations=24,
              central_root=False, mode_files=None):
    if not isinstance(graph, Hypergraph):
        raise TypeError("graph must be a Hypergraph")
    capacity(graph.n, epsilon)
    if not isinstance(central_root, bool):
        raise ValueError('central_root must be a boolean')
    if central_root and epsilon == 0:
        raise ValueError('the central root exception applies to imbalance-band runs')
    if mode_files is not None:
        if isinstance(mode_files, (str, bytes)) or len(mode_files) != 2:
            raise ValueError('mode_files must contain unscaled and normalized cache paths')
        mode_files=tuple(str(path) for path in mode_files)
    for name, value, minimum in [('seed',seed,0), ('steps',steps,4), ('cold',cold,0), ('replicas',replicas,1), ('root_iterations',root_iterations,1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if backend == 'reference':
        if central_root or mode_files is not None:
            raise ValueError('mode caches and the central root exception require the native backend')
        if device != 'cpu':
            raise ValueError("the portable reference uses CPU; request backend='native' for CUDA")
        from .reference import partition_reference
        return partition_reference(graph, epsilon, seed, steps, cold, replicas, root_iterations)
    if backend == 'educational':
        if device != 'cpu':raise ValueError('the educational Adam solver uses CPU')
        from .educational import partition_reference
        result=partition_reference(graph,epsilon,seed,steps,cold,replicas)
        result['backend']='educational'
        return result
    if backend != 'native':
        raise ValueError("backend must be 'reference', 'educational' or 'native'")
    if replicas != 48:
        raise ValueError("the paper solver uses the fixed 48-replica portfolio")
    try:
        engine = importlib.import_module('fem_hypergraph_partition._engine.entry')
    except ModuleNotFoundError as exc:
        if not exc.name.startswith('fem_hypergraph_partition._engine'):
            raise
        raise RuntimeError("Install a matching compiled backend wheel for the full catalytic solver. The catalyst-off baseline requires backend='reference' explicitly.") from exc
    if not callable(getattr(engine, 'protocol', None)) or engine.protocol().get('name') != 'paper-qubo-v1':
        raise RuntimeError("This compiled engine predates the paper policy. Install a native wheel at version 0.1.0a4 or newer.")
    result = engine.partition(graph.n, graph.edges, epsilon, seed, steps, cold, device, capture,
                              root_iterations, central_root, mode_files)
    # Always recompute the advertised cut and imbalance outside the engine.
    measured = score(graph, result['partition'], epsilon)
    for key in ['cut', 'block_sizes', 'feasible']:
        if result[key] != measured[key]:
            raise RuntimeError("compiled backend output failed independent scoring")
    selected = score(graph, result['selected_partition'], epsilon)
    if result['selected_cut'] != selected['cut'] or result['selected_feasible'] != selected['feasible']:
        raise RuntimeError("compiled backend descent output failed independent scoring")
    result.update(measured)
    return result
