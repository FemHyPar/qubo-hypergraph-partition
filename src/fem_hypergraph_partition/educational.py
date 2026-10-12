"""Portable independent-Bernoulli FEM baseline, without private catalysts.

This educational baseline is not the solver used for the manuscript tables.
"""
import time
import numpy as np
from .model import CutNetQUBO, score


def partition_reference(graph, epsilon, seed, steps, cold, replicas):
    started = time.perf_counter()
    model = CutNetQUBO(graph, epsilon)
    h = np.random.default_rng(seed).normal(0, .01, (replicas, model.variables))
    first = h.copy()
    a = np.zeros_like(h); b = np.zeros_like(h)
    temperatures = np.r_[np.geomspace(8., .02, steps), np.full(cold, .02)]
    penalties = np.r_[np.geomspace(.01, model.penalty, steps), np.full(cold, model.penalty)]
    best = None
    for t, (temperature, penalty) in enumerate(zip(temperatures, penalties), 1):
        p = 1/(1+np.exp(-h)); x = p[:, :model.n]
        u = p[:, model.n:model.n+model.m]; v = p[:, model.n+model.m:model.n+2*model.m]
        sums = np.stack([np.bincount(model.ids, weights=row[model.pins], minlength=model.m) for row in x])
        field = np.stack([np.bincount(model.pins, weights=row[model.ids], minlength=model.n) for row in v-u])
        weights = np.r_[np.ones(model.n), np.zeros(2*model.m), -model.slack_weights]
        mean = p@weights-model.lower
        balance = penalty*(2*mean[:, None]*weights+weights**2*(1-2*p))
        cost = np.c_[field, model.sizes-1-sums, sums-1, np.zeros((replicas, len(model.slack_weights)))]
        gradient = p*(1-p)*(cost+balance+temperature*h)/max(1, model.penalty)
        a = .9*a+.1*gradient; b = .999*b+.001*gradient**2
        h -= .08*(a/(1-.9**t))/(np.sqrt(b/(1-.999**t))+1e-12)
        np.clip(h, -20, 20, out=h)
        if t % 32 == 0 or t == len(temperatures):
            for r in range(replicas):
                bits = model.encode(h[r, :model.n] > 0)
                measured = score(graph, bits[:model.n], epsilon)
                key = (not measured['feasible'], model.energy(bits))
                if best is None or key < best[0]:
                    best = (key, bits[:model.n].copy(), measured)
    return dict(partition=best[1].tolist(), **best[2], backend='reference',
                seconds=time.perf_counter()-started, seed=seed, replicas=replicas,
                updates=steps+cold, initialization='original-graph random fields',
                initial_fields_sha256=__import__('hashlib').sha256(first.tobytes()).hexdigest(),
                catalyst=False, manuscript_backend=False)
