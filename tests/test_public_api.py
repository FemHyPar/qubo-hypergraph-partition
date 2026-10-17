from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from fem_hypergraph_partition import Hypergraph, partition, score
from fem_hypergraph_partition.model import CutNetQUBO


@pytest.mark.parametrize('epsilon', [0, .03, .4])
def test_qubo_truth_table(epsilon):
    graph = Hypergraph(5, ((0,1,2),(2,3),(4,)))
    model = CutNetQUBO(graph, epsilon)
    energies = []
    for state in product([0,1], repeat=model.variables):
        energies.append((model.energy(state), state))
    minimum = min(e for e, _ in energies)
    assert all(score(graph, state[:5], epsilon)['feasible'] for e, state in energies if e == minimum)
    for x in product([0,1], repeat=5):
        encoded = model.encode(x)
        actual = score(graph, x, epsilon)
        distance = max(model.lower-sum(x), 0, sum(x)-model.upper)
        penalty=(sum(x)-model.lower)*(sum(x)-model.upper) if epsilon==0 else distance**2
        assert model.energy(encoded) == actual['cut']+model.penalty*penalty
        tail_min = min(model.energy((*x,*tail)) for tail in product([0,1], repeat=model.variables-5))
        assert tail_min == model.energy(encoded)


def test_fp32_protocol_requires_matching_engine(monkeypatch):
    import fem_hypergraph_partition.api as api
    monkeypatch.setattr(api.importlib,'import_module',lambda name:SimpleNamespace())
    with pytest.raises(RuntimeError,match='FP32 paper protocol'):
        api.partition_fp32(Hypergraph(6,((0,1,2),(3,4,5),(2,3))))


def test_compiled_fp32_policy_when_installed():
    engine=pytest.importorskip('fem_hypergraph_partition._engine.entry')
    if not callable(getattr(engine,'protocol_fp32',None)):pytest.skip('FP32 backend not installed')
    from fem_hypergraph_partition import partition_fp32
    graph=Hypergraph(24,tuple(tuple((i+j)%24 for j in range(3)) for i in range(24)))
    for epsilon in [0.,.03]:
        result=partition_fp32(graph,epsilon=epsilon,steps=64,cold=64,capture=False)
        assert result['precision']=='float32' and result['replicas']==1
        assert result['protocol']['name']=='paper-qubo-fp32-v2'
        assert all(result[k]==v for k,v in score(graph,result['partition'],epsilon).items())


def test_readout_is_independently_scored_and_reproducible():
    graph = Hypergraph(6, ((0,1,2),(3,4,5),(2,3)))
    options = dict(seed=4, steps=64, cold=16, replicas=4, epsilon=0, backend='reference')
    a = partition(graph, **options); b = partition(graph, **options)
    assert a['partition'] == b['partition']
    assert all(a[k] == v for k,v in score(graph,a['partition'],0).items())
    assert a['initialization'] == 'original-graph random fields'


def test_validation_rejects_lossy_inputs(tmp_path):
    with pytest.raises(ValueError): Hypergraph(4, ((0,.5),))
    with pytest.raises(ValueError): score(Hypergraph(4,((0,1),)), [0,0,0,2])
    p = tmp_path/'weighted.hgr'; p.write_text('1 4 1\n2 1 2\n')
    with pytest.raises(ValueError, match='unweighted'): Hypergraph.read_hgr(p)
    with pytest.raises(ValueError): partition(Hypergraph(4,((0,1),)), steps=0)


def test_default_requires_the_complete_compiled_policy(monkeypatch):
    import fem_hypergraph_partition.api as api
    graph=Hypergraph(6,((0,1,2),(3,4,5),(2,3)))
    monkeypatch.setattr(api.importlib,'import_module',lambda name:SimpleNamespace())
    with pytest.raises(RuntimeError,match='predates the paper policy'):
        partition(graph)


def test_default_preserves_full_budget_and_reports_raw_endpoint(monkeypatch):
    import fem_hypergraph_partition.api as api
    graph=Hypergraph(6,((0,1,2),(3,4,5),(2,3)))
    calls=[]
    def run(*args):
        calls.append(args)
        x=[0,0,0,1,1,1];measured=score(graph,x,.03)
        return dict(partition=x,**measured,selected_partition=x,
                    selected_cut=measured['cut'],selected_feasible=measured['feasible'])
    engine=SimpleNamespace(protocol=lambda:dict(name='paper-qubo-v1'),partition=run)
    monkeypatch.setattr(api.importlib,'import_module',lambda name:engine)
    result=partition(graph)
    assert calls[0][3:]==(21,12000,144,'cpu',True,24,False,None)
    assert result['partition']==result['selected_partition']


def test_complete_compiled_policy_when_installed():
    pytest.importorskip('fem_hypergraph_partition._engine.entry')
    edges=tuple(tuple(12*g+(i+j)%12 for j in range(3))
                for g in range(2) for i in range(12))+((0,12),(5,17))
    graph=Hypergraph(24,edges)
    for epsilon in (0,.03):
        result=partition(graph,epsilon=epsilon,steps=64,cold=32,capture=False)
        assert result['replicas']==48 and result['updates']==96
        assert result['precision']=='float64' and result['root_iterations']==24
        assert result['protocol']['name']=='paper-qubo-v1'
        assert result['catalyst'] and result['primary_endpoint']=='raw'
        assert all(result[k]==v for k,v in score(graph,result['partition'],epsilon).items())
