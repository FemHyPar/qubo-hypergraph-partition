import argparse
import json
from pathlib import Path
from . import Hypergraph, partition


def main():
    p = argparse.ArgumentParser(description='Original-graph QUBO hypergraph partitioning')
    p.add_argument('hgr', type=Path)
    p.add_argument('--epsilon', type=float, default=.03)
    p.add_argument('--backend', choices=['native','reference','educational'], default='native')
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--cold',type=int,default=144)
    p.add_argument('--replicas',type=int,default=48)
    p.add_argument('--root-iterations',type=int,default=24)
    p.add_argument('--central-root',action='store_true',help='Use the explicitly requested central root safeguard')
    p.add_argument('--modes',type=Path,nargs=2,metavar=('UNSCALED','NORMALIZED'),help='Certified original-incidence mode caches')
    p.add_argument('--capture',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = partition(Hypergraph.read_hgr(args.hgr), epsilon=args.epsilon,
                       backend=args.backend, device=args.device, steps=args.steps, seed=args.seed,
                       cold=args.cold,replicas=args.replicas,capture=args.capture,root_iterations=args.root_iterations,
                       central_root=args.central_root,mode_files=args.modes)
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
