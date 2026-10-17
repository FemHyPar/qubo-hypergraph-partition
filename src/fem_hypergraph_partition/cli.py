import argparse
import json
from pathlib import Path
from . import Hypergraph, partition,partition_fp32


def main():
    p = argparse.ArgumentParser(description='Original-graph QUBO hypergraph partitioning')
    p.add_argument('hgr', type=Path)
    p.add_argument('--epsilon', type=float, default=.03)
    p.add_argument('--backend', choices=['native','reference','educational'], default='native')
    p.add_argument('--device', default='cpu')
    p.add_argument('--protocol',choices=['fp32','legacy'],default='fp32')
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--cold',type=int,default=144)
    p.add_argument('--replicas',type=int)
    p.add_argument('--root-iterations',type=int)
    p.add_argument('--replica',type=int,help='Explicit preselected profile index in [0,48)')
    p.add_argument('--arm',help='Explicit catalyst-channel policy')
    p.add_argument('--central-root',action='store_true',help='Use the explicitly requested central root safeguard')
    p.add_argument('--modes',type=Path,nargs=2,metavar=('UNSCALED','NORMALIZED'),help='Certified original-incidence mode caches')
    p.add_argument('--capture',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.protocol=='fp32' and args.backend=='native':
        if args.replicas not in (None,1):p.error('The FP32 paper protocol uses batch=1')
        result=partition_fp32(args.hgr,epsilon=args.epsilon,seed=args.seed,steps=args.steps if args.steps is not None else 3000,
            cold=args.cold,device=args.device,capture=args.capture,root_iterations=args.root_iterations if args.root_iterations is not None else 64,
            replica=args.replica,arm=args.arm,mode_files=args.modes)
    else:
        result = partition(Hypergraph.read_hgr(args.hgr), epsilon=args.epsilon,
                       backend=args.backend, device=args.device, steps=args.steps if args.steps is not None else 12000, seed=args.seed,
                       cold=args.cold,replicas=args.replicas if args.replicas is not None else 48,capture=args.capture,root_iterations=args.root_iterations if args.root_iterations is not None else 24,
                       central_root=args.central_root,mode_files=args.modes)
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
