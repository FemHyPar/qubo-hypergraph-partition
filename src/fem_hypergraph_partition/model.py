"""Unweighted two-way cut-net inputs, independent scoring and exact QUBO."""
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import math
import numpy as np


@dataclass(frozen=True)
class Hypergraph:
    n: int
    edges: tuple

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 2:
            raise ValueError("n must be an integer of at least 2")
        normalized = []
        for edge in self.edges:
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in edge):
                raise ValueError("vertex IDs must be integers")
            row = tuple(int(v) for v in edge)
            if not row or len(set(row)) != len(row) or min(row) < 0 or max(row) >= self.n:
                raise ValueError("each edge needs distinct vertex IDs in [0,n)")
            normalized.append(row)
        if not normalized:
            raise ValueError("at least one hyperedge is required")
        object.__setattr__(self, "edges", tuple(normalized))

    @cached_property
    def _incidence(self):
        sizes = np.fromiter(map(len, self.edges), dtype=np.int64, count=len(self.edges))
        ptr = np.r_[0, np.cumsum(sizes)]
        pins = np.fromiter((v for e in self.edges for v in e), dtype=np.int64, count=int(ptr[-1]))
        for array in (sizes, ptr, pins):
            array.flags.writeable = False
        return sizes, ptr, pins

    @classmethod
    def read_hgr(cls, path):
        # Format 0 only. Preserve singleton nets and isolated vertices.
        lines = [s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith("%")]
        if not lines:
            raise ValueError("empty .hgr file")
        header = list(map(int, lines[0].split()))
        if len(header) not in (2, 3) or (len(header) == 3 and header[2] != 0):
            raise ValueError("only unweighted .hgr format 0 is supported")
        m, n = header[:2]
        if len(lines) != m + 1:
            raise ValueError("hyperedge count differs from header")
        return cls(n, tuple(tuple(int(v)-1 for v in line.split()) for line in lines[1:]))


def capacity(n, epsilon):
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must be a finite fraction in [0,1)")
    upper = min(n, math.floor((1+epsilon)*math.ceil(n/2)))
    return n-upper, upper


def binary_vector(values, size):
    x = np.asarray(values)
    if x.shape != (size,) or not np.isin(x, [0, 1]).all():
        raise ValueError("partition must contain exactly n binary labels")
    return x.astype(np.int8)


def score(graph, partition, epsilon=0.03):
    x = binary_vector(partition, graph.n)
    lower, upper = capacity(graph.n, epsilon)
    ones = int(x.sum())
    sizes = [graph.n-ones, ones]
    net_sizes, ptr, pins = graph._incidence
    counts = np.add.reduceat(x[pins].astype(np.int64), ptr[:-1])
    cut = int(np.count_nonzero((counts > 0) & (counts < net_sizes)))
    return dict(cut=int(cut), block_sizes=sizes,
                observed_imbalance_percent=100*(max(sizes)/math.ceil(graph.n/2)-1),
                feasible=lower <= ones <= upper, lower=lower, upper=upper)


class CutNetQUBO:
    """Two auxiliaries per net, plus bounded count-slack bits.

    min_{u,v} [1+(r-1-k)u+(k-1)v] = 1[0<k<r].
    Balance costs lambda*(sum(x)-lower-sum(w*s))**2.
    lambda > maximum incidence degree makes global minima feasible.
    """
    def __init__(self, graph, epsilon=0.03):
        self.graph = graph
        self.n, self.m = graph.n, len(graph.edges)
        self.lower, self.upper = capacity(graph.n, epsilon)
        self.sizes = np.array(list(map(len, graph.edges)))
        self.pins = np.concatenate(graph.edges).astype(np.int64)
        self.ids = np.repeat(np.arange(self.m), self.sizes)
        width = self.upper-self.lower
        count = width.bit_length()
        self.slack_weights = np.r_[2**np.arange(count-1), width-(2**(count-1)-1)].astype(np.int64) if count else np.zeros(0, dtype=np.int64)
        self.penalty = int(np.bincount(self.pins, minlength=self.n).max())+1
        self.variables = self.n+2*self.m+count

    def encode(self, partition):
        x = binary_vector(partition, self.n)
        counts = np.bincount(self.ids, weights=x[self.pins], minlength=self.m)
        value = int(np.clip(int(x.sum())-self.lower, 0, self.upper-self.lower))
        bits = np.zeros(len(self.slack_weights), dtype=np.int8)
        for i in range(len(bits)-1, -1, -1):
            if value >= self.slack_weights[i]:
                bits[i] = 1
                value -= self.slack_weights[i]
        assert value == 0
        return np.r_[x, counts == self.sizes, counts == 0, bits].astype(np.int8)

    def energy(self, bits):
        b = binary_vector(bits, self.variables)
        x, u, v = b[:self.n], b[self.n:self.n+self.m], b[self.n+self.m:self.n+2*self.m]
        counts = np.bincount(self.ids, weights=x[self.pins], minlength=self.m)
        net = (1+(self.sizes-1-counts)*u+(counts-1)*v).sum()
        residual = int(x.sum())-self.lower-int(b[self.n+2*self.m:]@self.slack_weights)
        return float(net+self.penalty*residual**2)
