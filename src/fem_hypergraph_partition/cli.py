import argparse
import json
from pathlib import Path
from . import Hypergraph, partition


def main():
    p = argparse.ArgumentParser(description='Original-graph QUBO hypergraph partitioning')
    p.add_argument('hgr', type=Path)
    p.add_argument('--epsilon', type=float, default=.03)
    p.add_argument('--backend', choices=['reference','native'], default='reference')
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = partition(Hypergraph.read_hgr(args.hgr), epsilon=args.epsilon,
                       backend=args.backend, device=args.device, steps=args.steps, seed=args.seed)
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
