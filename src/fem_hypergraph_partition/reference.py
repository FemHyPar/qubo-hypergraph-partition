"""Portable manuscript base FEM, with every catalyst and local offset off.

Exact auxiliary Gibbs blocks, a correlated slack block and implicit count
updates follow the Methods. This is a CPU mechanism reference, not GPU code.
"""
import hashlib
import time
import numpy as np
from .model import CutNetQUBO, score


def auxiliary_blocks(spin_sums, sizes, temperature):
    count=(spin_sums+sizes)/2
    energies=np.stack([np.zeros_like(count),sizes-1-count,count-1,
        np.broadcast_to(sizes-2,count.shape)],axis=-1)
    logp=-energies/temperature;logp-=logp.max(axis=-1,keepdims=True)
    probability=np.exp(logp);probability/=probability.sum(axis=-1,keepdims=True)
    return probability


def auxiliary_difference(spin_sums,sizes,temperature):
    # Exact Gibbs response, centered to avoid cancelling nearly equal means.
    a=-(sizes/2-1)/temperature;b=spin_sums/(2*temperature)
    logcosh=lambda x:np.abs(x)+np.logaddexp(0,-2*np.abs(x))-np.log(2.)
    logratio=logcosh(b)-logcosh(a)
    return -np.tanh(b)*np.exp(-np.logaddexp(0,-logratio))


def implicit_update(h,cost,vertex_step,penalty,temperature,target,iterations=24):
    spin=np.tanh(h/2);step=vertex_step/(1+2*penalty*vertex_step)
    rate=step/(1+step*temperature)
    tilted=np.clip((h-step*(cost-penalty*spin))/(1+step*temperature),-20,20)
    kappa=2*penalty*rate;target=np.asarray(target).reshape(-1,1)
    crossings=(tilted-np.log(target/(h.shape[-1]-target)))/kappa
    low=np.minimum(crossings.min(-1,keepdims=True),0)
    high=np.maximum(crossings.max(-1,keepdims=True),0);delta=np.zeros_like(low)
    for _ in range(iterations):
        m=np.tanh((tilted-kappa*delta)/2)
        value=delta-(h.shape[-1]/2-target+.5*m.sum(-1,keepdims=True))
        derivative=1+.25*(kappa*(1-m*m)).sum(-1,keepdims=True)
        low=np.where(value<0,delta,low);high=np.where(value>0,delta,high)
        proposed=delta-value/derivative
        delta=np.where((proposed>=low)&(proposed<=high),proposed,(low+high)/2)
    updated=tilted-kappa*delta
    residual=delta-(h.shape[-1]/2-target+.5*np.tanh(updated/2).sum(-1,keepdims=True))
    return updated,residual


class BaseFEM:
    def __init__(self,graph,epsilon=0.,seed=21,steps=12000,cold=144,replicas=48,root_iterations=24):
        self.model=CutNetQUBO(graph,epsilon);m=self.model
        self.steps,self.cold,self.replicas,self.root_iterations=steps,cold,replicas,root_iterations
        shared=np.random.default_rng(seed).normal(0,.01,(1,m.n))
        noise=np.random.default_rng(np.random.SeedSequence([seed,20261006])).normal(0,.001,(replicas,m.n))
        nearby=shared+noise
        self.initial=nearby[np.arange(replicas)%min(16,replicas)].copy();self.h=self.initial.copy()
        rates=np.array([1.]*12+[.5,.5,1.,1.]);degree=np.bincount(m.pins,minlength=m.n)
        self.vertex_step=rates[np.arange(replicas)%16,None]/(1+degree[None,:])
        self.temperature=np.r_[np.geomspace(8.,.02,steps),np.full(cold,.02)]
        ramp=max(2,int(round(steps*.1)))
        self.penalty=np.r_[np.full(steps-ramp,.01),np.geomspace(.01,m.penalty,ramp),np.full(cold,m.penalty)]
        width=m.upper-m.lower;mult=np.ones(1,dtype=np.int64)
        for weight in m.slack_weights:
            expanded=np.zeros(len(mult)+weight,dtype=np.int64)
            expanded[:len(mult)]+=mult;expanded[weight:]+=mult;mult=expanded
        self.slack_values=np.arange(width+1,dtype=float);self.log_multiplicity=np.log(mult)
        self.last_slack=None;self.max_root_residual=0.;self.index=0

    def step(self):
        m=self.model;t=self.temperature[self.index];penalty=self.penalty[self.index]
        spin=np.tanh(self.h/2)
        sums=np.stack([np.bincount(m.ids,weights=row[m.pins],minlength=m.m) for row in spin])
        difference=auxiliary_difference(sums,m.sizes,t)
        cost=np.stack([np.bincount(m.pins,weights=row[m.ids],minlength=m.n) for row in difference])
        mean=m.n/2+.5*spin.sum(-1,keepdims=True)
        if len(m.slack_weights):
            logp=self.log_multiplicity[None,:]-penalty*(mean-m.lower-self.slack_values[None,:])**2/t
            logp-=logp.max(-1,keepdims=True);q=np.exp(logp);q/=q.sum(-1,keepdims=True)
            self.last_slack=q;target=m.lower+(q*self.slack_values).sum(-1,keepdims=True)
        else:target=np.full((self.replicas,1),m.n/2 if m.strict_symmetric else m.lower)
        self.h,residual=implicit_update(self.h,cost,self.vertex_step,penalty,t,target,self.root_iterations)
        self.max_root_residual=max(self.max_root_residual,float(np.abs(residual).max()));self.index+=1
        if self.max_root_residual>=max(1e-7,1e-11*m.n):
            raise RuntimeError('Implicit root did not converge; retain the failure and try root_iterations=64.')

    def checkpoint(self):
        bits=np.stack([self.model.encode(row>0) for row in self.h])
        return bits,np.array([self.model.energy(row) for row in bits])


def one_bit_descent(model,initial,max_steps=50000):
    """Full original polynomial descent; no projection or move filtering."""
    bits=initial.copy().astype(np.int64);initial_energy=model.energy(bits)
    flips=0;started=time.perf_counter()
    for _ in range(max_steps+1):
        x,u,v=bits[:model.n],bits[model.n:model.n+model.m],bits[model.n+model.m:model.n+2*model.m]
        counts=np.bincount(model.ids,weights=x[model.pins],minlength=model.m)
        residual=int(x.sum())-model.lower-int(bits[model.n+2*model.m:]@model.slack_weights)
        field=np.bincount(model.pins,weights=(v-u)[model.ids],minlength=model.n)
        slack=bits[model.n+2*model.m:]
        center=(2*int(x.sum())-model.lower-model.upper) if model.strict_symmetric else 2*residual
        delta=np.r_[field*(1-2*x)+model.penalty*(center*(1-2*x)+1),
            (model.sizes-1-counts)*(1-2*u),(counts-1)*(1-2*v),
            model.penalty*(-2*residual*model.slack_weights*(1-2*slack)+model.slack_weights**2)]
        i=int(delta.argmin());minimum=float(delta[i])
        if minimum>=-1e-12:break
        if flips==max_steps:raise RuntimeError('Original-QUBO descent reached its limit')
        bits[i]^=1;flips+=1
    assert model.energy(bits)<=initial_energy
    return bits.astype(np.int8),dict(flips=flips,minimum_remaining_delta=minimum,seconds=time.perf_counter()-started)


def partition_reference(graph,epsilon,seed,steps,cold,replicas,root_iterations=24):
    started=time.perf_counter();runner=BaseFEM(graph,epsilon,seed,steps,cold,replicas,root_iterations)
    best,energy=runner.checkpoint();checkpoints=[]
    planned=set(range(128,steps+cold+1,128))|{steps,steps+cold}
    for step in range(1,steps+cold+1):
        runner.step()
        if step in planned:
            bits,value=runner.checkpoint();better=value<energy
            best[better]=bits[better];energy=np.minimum(energy,value);checkpoints.append(step)
    rows=[];selected=[]
    for index,bits in enumerate(best):
        final,descent=one_bit_descent(runner.model,bits);measured=score(graph,final[:graph.n],epsilon)
        assert measured['feasible']
        selected.append(final);rows.append(dict(replica=index,raw=score(graph,bits[:graph.n],epsilon),
            raw_energy=runner.model.energy(bits),selected=measured,selected_energy=runner.model.energy(final),descent=descent))
    winner=min(range(replicas),key=lambda i:rows[i]['selected_energy'])
    return dict(partition=selected[winner][:graph.n].tolist(),**rows[winner]['selected'],backend='reference',
        seconds=time.perf_counter()-started,seed=seed,replicas=replicas,updates=steps+cold,
        initialization='original-graph random fields',initial_fields_sha256=hashlib.sha256(runner.initial.tobytes()).hexdigest(),
        catalyst=False,manuscript_base=True,manuscript_backend=False,root_iterations=root_iterations,
        max_root_residual=runner.max_root_residual,checkpoints=sorted(checkpoints),replica_results=rows,
        scope='Portable catalyst-off base FEM; no accelerated or catalytic performance claim.')
