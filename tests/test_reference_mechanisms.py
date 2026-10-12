"""Independent checks of the published base equations, without catalysts."""
from itertools import product
import numpy as np
import pytest
from fem_hypergraph_partition import Hypergraph,partition
from fem_hypergraph_partition.model import CutNetQUBO
from fem_hypergraph_partition.reference import (BaseFEM,auxiliary_blocks,
    auxiliary_difference,implicit_update,one_bit_descent)


def test_four_state_gibbs_response_and_entropy():
    sums=np.array([[1e-14,-1.2,2.3],[-.7,.8,-2.]])
    sizes=np.array([1,3,8]);q=auxiliary_blocks(sums,sizes,.7)
    np.testing.assert_allclose(q.sum(-1),1,atol=1e-15)
    np.testing.assert_allclose(q[...,2]-q[...,1],auxiliary_difference(sums,sizes,.7),atol=5e-16)
    count=(sums+sizes)/2
    energy=np.stack([np.zeros_like(count),sizes-1-count,count-1,np.broadcast_to(sizes-2,count.shape)],-1)
    entropy=np.sum(np.where(q>0,q*np.log(q),0),-1)
    shifted=-energy/.7
    logz=shifted.max(-1)+np.log(np.exp(shifted-shifted.max(-1,keepdims=True)).sum(-1))
    np.testing.assert_allclose((q*energy).sum(-1)+.7*entropy,-.7*logz,atol=2e-15)


def test_correlated_slack_matches_binary_enumeration():
    graph=Hypergraph(12,((0,1,2),(4,5),(8,9,10,11)))
    runner=BaseFEM(graph,epsilon=.5,steps=32,cold=0,replicas=3)
    runner.step();m=runner.model
    assignments=np.array(list(product([0,1],repeat=len(m.slack_weights))))
    values=assignments@m.slack_weights
    # The first target was calculated at these initial vertex marginals.
    means=graph.n/2+.5*np.tanh(runner.initial/2).sum(-1)
    for i,mean in enumerate(means):
        prob=np.exp(-.01*(mean-m.lower-values)**2/8);prob/=prob.sum()
        expected=np.bincount(values,weights=prob,minlength=m.upper-m.lower+1)
        np.testing.assert_allclose(runner.last_slack[i],expected,atol=2e-16)


@pytest.mark.parametrize('target',[7.,8.,8.4])
def test_implicit_balance_matches_independent_bisection(target):
    rng=np.random.default_rng(314);h=rng.normal(size=(2,16));cost=rng.normal(size=h.shape)
    d=np.full_like(h,.3);penalty=13.;temperature=.08
    actual,residual=implicit_update(h,cost,d,penalty,temperature,np.full((2,1),target),64)
    r=d/(1+2*penalty*d);rate=r/(1+r*temperature)
    tilted=np.clip((h-r*(cost-penalty*np.tanh(h/2)))/(1+r*temperature),-20,20)
    for i in range(2):
        low,high=-100.,100.
        for _ in range(160):
            delta=(low+high)/2
            f=delta-(1/(1+np.exp(-(tilted[i]-2*penalty*rate[i]*delta)))).sum()+target
            if f<0:low=delta
            else:high=delta
        np.testing.assert_allclose(actual[i],tilted[i]-2*penalty*rate[i]*delta,atol=2e-14)
    assert np.abs(residual).max()<2e-14


def test_every_readout_is_an_original_qubo_single_bit_minimum():
    graph=Hypergraph(6,((0,1,2),(2,3),(3,4,5),(0,)))
    for epsilon in [0.,.4]:
        m=CutNetQUBO(graph,epsilon)
        for state in np.random.default_rng(71).integers(0,2,size=(24,m.variables),dtype=np.int8):
            final,info=one_bit_descent(m,state)
            for i in range(m.variables):
                changed=final.copy();changed[i]^=1
                assert m.energy(changed)>=m.energy(final)
            assert info['minimum_remaining_delta']>=0


def test_primary_checkpoint_rules_and_replica_selection():
    graph=Hypergraph(6,((0,1,2),(3,4,5),(2,3)))
    result=partition(graph,epsilon=0,steps=132,cold=8,replicas=4)
    assert result['checkpoints']==[128,132,140]
    assert len(result['replica_results'])==4 and result['manuscript_base']
    assert result['cut']==min(r['selected']['cut'] for r in result['replica_results'])
