"""Exhaustive checks of the paper's two 12-vertex counterexamples."""
from itertools import combinations
from pathlib import Path
import numpy as np
from fem_hypergraph_partition import Hypergraph,score
from fem_hypergraph_partition.model import CutNetQUBO


def test_two_paper_counterexamples_have_exact_objectives_and_distorted_surrogates():
    examples=Path(__file__).resolve().parents[1]/'examples'
    for name,minimum,surrogate,wrong_cut in [('counterexample_a.hgr',7,'unit',8),('counterexample_b.hgr',5,'normalized',6)]:
        graph=Hypergraph.read_hgr(examples/name);m=CutNetQUBO(graph,0);rows=[]
        for ones in combinations(range(graph.n),graph.n//2):
            x=np.zeros(graph.n,dtype=np.int8);x[list(ones)]=1
            native=score(graph,x,0)['cut'];assert m.energy(m.encode(x))==native
            counts=np.bincount(m.ids,weights=x[m.pins],minlength=m.m)
            pairs=counts*(m.sizes-counts)
            rows.append((native,float(pairs.sum()),float((pairs/np.maximum(1,m.sizes-1)).sum())))
        assert len(rows)==924 and min(r[0] for r in rows)==minimum
        column=1 if surrogate=='unit' else 2;best=min(r[column] for r in rows)
        assert all(r[0]==wrong_cut for r in rows if abs(r[column]-best)<1e-12)
