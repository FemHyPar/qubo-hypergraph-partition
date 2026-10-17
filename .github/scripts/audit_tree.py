"""Verify the exact SDK manifest and reject protected files in its ancestry."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
ALLOWED={'.gitignore','README.md','LICENSE','NOTICE','pyproject.toml',
    'examples/tiny.hgr','examples/counterexample_a.hgr','examples/counterexample_b.hgr',
    'tests/test_public_api.py','tests/test_reference_mechanisms.py','tests/test_paper_examples.py',
    'docs/reproducibility.md','docs/plot_benchmarks.py','docs/check_benchmarks.py',
    'docs/benchmarks/paper_tables.json','docs/benchmarks/base_validation.json','docs/benchmarks/native_artifact.json',
    'paper/manuscript.pdf','paper/supplementary.pdf',
    'docs/benchmarks/warm_process_fp32_2026-10-17.json','docs/images/runtime_breakdown.png',
    'docs/benchmarks/paper_tables_before_native_cuda_2026-10-17.json',
    'docs/benchmarks/fp32_primary_2026-10-17.json','docs/benchmarks/paper_tables_before_fp32_primary_2026-10-17.json','docs/benchmarks/historical_controls_2026-10-16.json','docs/benchmarks/time_quality_2026-10-16.json','docs/benchmarks/time_to_target_2026-10-17.json','docs/benchmarks/time_to_target_priority_fixed_2026-10-17.json','docs/images/frozen_workflow.png','docs/images/time_quality.png','docs/images/time_quality_aggregate.png','docs/images/time_quality_aggregate_e01.png','docs/images/time_quality_aggregate_e03.png','docs/images/time_quality_e01.png','docs/images/time_quality_e03.png','docs/images/time_quality_scircuit.png',
    '.github/scripts/audit_tree.py','.github/workflows/sync-public-release.yml',
    *('docs/images/'+name+'.png' for name in ['method','objectives','qubo_quality','qubo_runtime']),
    *('src/fem_hypergraph_partition/'+name+'.py' for name in ['__init__','api','model','reference','educational','cli'])}


def main():
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    paths={p for p in tracked if p};assert paths==set(manifest)|{'release_manifest.json'}
    assert set(manifest)<=ALLOWED and sum((ROOT/p).stat().st_size for p in manifest)<30_000_000
    for path,digest in manifest.items():
        f=ROOT/path;assert not f.is_symlink()
        assert hashlib.sha256(f.read_bytes()).hexdigest()==digest,path
    # Only objects reachable from this release head are inspected. Other
    # private branches in the upstream must never be merged into this branch.
    objects=subprocess.check_output(['git','rev-list','--objects','HEAD'],cwd=ROOT).decode().splitlines()
    forbidden=['experiments/','tools/','src/fem/','private_release/','doc/main.tex','_engine/']
    suffixes={'.cu','.cuh','.c','.cpp','.pyx','.pxd','.ptx','.so','.cubin','.npy','.npz','.whl'}
    for line in objects:
        parts=line.split(' ',1)
        if len(parts)==2:
            path=parts[1]
            assert not any(path.startswith(prefix) or prefix=='_engine/' and prefix in path for prefix in forbidden),path
            assert Path(path).suffix not in suffixes,path
    print(json.dumps(dict(passed=True,manifest_files=len(manifest),reachable_objects=len(objects),protected_history_absent=True)))


if __name__=='__main__':main()
