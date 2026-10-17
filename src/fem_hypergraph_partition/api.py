"""Stable release boundary; backend selection is explicit."""
import importlib
import hashlib
import json
from pathlib import Path
from .model import Hypergraph, capacity, score


def partition_fp32(graph, *, epsilon=.03, seed=21, steps=3000, cold=144,
                   device='cpu', capture=True, root_iterations=64,
                   replica=None, arm=None, mode_files=None):
    """Current paper solver; graph may be a Hypergraph or an unweighted .hgr path.

    A path uses the compiled input reader. Original paper inputs use their
    frozen instance/capacity policy; other inputs default to profile 7 with
    all channels. Explicit replica/arm overrides are reported in the result.
    """
    path=Path(graph) if isinstance(graph,(str,Path)) else None
    if path is None and not isinstance(graph,Hypergraph):raise TypeError('Expected a Hypergraph or .hgr path')
    if path is not None and not path.is_file():raise FileNotFoundError(path)
    capacity(graph.n if path is None else 2,epsilon)
    for name,value,minimum in [('seed',seed,0),('steps',steps,4),('cold',cold,0),('root_iterations',root_iterations,1)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError(f'Invalid {name}')
    try:engine=importlib.import_module('fem_hypergraph_partition._engine.entry')
    except ModuleNotFoundError as exc:
        if not exc.name.startswith('fem_hypergraph_partition._engine'):raise
        raise RuntimeError('Install the matching compiled backend wheel at version 0.1.0a5 or newer') from exc
    if not callable(getattr(engine,'protocol_fp32',None)) or engine.protocol_fp32().get('name')!='paper-qubo-fp32-v2':
        raise RuntimeError('This backend lacks the current FP32 paper protocol; install version 0.1.0a5 or newer')
    choice={}
    if path is not None:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        policies=Path(engine.__file__).parent/'policies.json'
        if policies.is_file():
            choice=next((r for r in json.loads(policies.read_text()) if r['input_sha256']==digest and r['epsilon']==epsilon),{})
    replica=choice.get('replica_index_zero_based',6) if replica is None else replica
    arm=choice.get('arm','full') if arm is None else arm
    if isinstance(replica,bool) or not isinstance(replica,int) or not 0<=replica<48:raise ValueError('replica must be in [0,48)')
    allowed={'full','off','pair_only','auxiliary_only','collective_only','temperature_only',
             'minus_pair','minus_auxiliary','minus_collective','minus_temperature'}
    if arm not in allowed:raise ValueError('Unknown channel policy')
    if mode_files is not None:
        if isinstance(mode_files,(str,bytes)) or len(mode_files)!=2:raise ValueError('Two mode-cache paths are required')
        mode_files=tuple(str(p) for p in mode_files)
    result=engine.partition_fp32(graph.n if path is None else 0,graph.edges if path is None else None,
        epsilon,seed,steps,cold,device,capture,root_iterations,replica,arm,mode_files,
        None if path is None else str(path))
    independent=graph if path is None else Hypergraph.read_hgr(path)
    measured=score(independent,result['partition'],epsilon)
    if result['cut']!=measured['cut'] or result['feasible']!=measured['feasible'] or result['block_sizes']!=measured['block_sizes']:
        raise RuntimeError('Compiled FP32 output failed independent integer scoring')
    if result['precision']!='float32' or result['replicas']!=1:raise RuntimeError('Unexpected paper protocol')
    result.update(measured);return result


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
