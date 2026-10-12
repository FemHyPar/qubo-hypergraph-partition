"""Validate user-supplied benchmark files against the paper's input hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from fem_hypergraph_partition import Hypergraph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True)
    a=p.parse_args();data=json.loads((Path(__file__).parent/'benchmarks/paper_tables.json').read_text())
    for row in data['qubo']:
        path=a.data_dir/row['instance']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['input_sha256']:
            raise ValueError(f'Input hash mismatch: {path.name}')
        graph=Hypergraph.read_hgr(path)
        if graph.n!=row['n'] or len(graph.edges)!=row['m']:raise ValueError(path.name)
    print(json.dumps(dict(passed=True,inputs_verified=len(data['qubo']))))


if __name__=='__main__':main()
