from itertools import product
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
        assert model.energy(encoded) == actual['cut']+model.penalty*distance**2
        tail_min = min(model.energy((*x,*tail)) for tail in product([0,1], repeat=model.variables-5))
        assert tail_min == model.energy(encoded)


def test_readout_is_independently_scored_and_reproducible():
    graph = Hypergraph(6, ((0,1,2),(3,4,5),(2,3)))
    options = dict(seed=4, steps=64, cold=16, replicas=4, epsilon=0)
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
