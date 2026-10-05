"""Matched forward cooperative bounds queries and CPU serial fallbacks."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from compare_spatial import ROOT, ITEMS, QUERIES, run, timed, exhaustive_sets


def make_kernel(wp, kind):
    @wp.kernel
    def candidates(tree: wp.uint64, params: wp.array(dtype=wp.float32),
                   output: wp.array(dtype=wp.float32)):
        q = wp.tid()
        p = wp.vec3(params[q * 7], params[q * 7 + 1], params[q * 7 + 2])
        d = wp.vec3(params[q * 7 + 3], params[q * 7 + 4], params[q * 7 + 5])
        if wp.static(kind == 0):
            radius = params[q * 7 + 6]
            query = wp.tile_bvh_query_aabb(tree, p - wp.vec3(radius), p + wp.vec3(radius))
        else:
            query = wp.tile_bvh_query_ray(tree, p, d)
        while wp.tile_query_valid(query):
            result = wp.tile_bvh_query_next(query)
            item = wp.untile(result)
            if item >= 0:
                wp.atomic_add(output, q * ITEMS + item, float(1.0))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warp-path', type=Path)
    parser.add_argument('--gpu-arch', default='sm_120')
    parser.add_argument('--work-dir', type=Path, default=Path('/tmp/bajo_warp_block'))
    args = parser.parse_args()
    if args.warp_path:
        sys.path.insert(0, str(args.warp_path))
    import warp as wp
    wp.config.kernel_cache_dir = '/tmp/bajo_warp_block_cache'
    wp.init()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    executable, prefix = work / 'block', work / 'fixture'
    run(['mojo', 'build', '-I', str(ROOT), '--target-accelerator', args.gpu_arch,
         str(ROOT / 'bench/warp/bench_block.mojo'), '-o', str(executable)])
    run([str(executable), 'fixtures', str(prefix)])
    boxes = np.fromfile(str(prefix) + '.boxes', dtype='<f4').reshape(ITEMS, 6)
    params = np.fromfile(str(prefix) + '.queries', dtype='<f4')
    data = dict(date=datetime.now(timezone.utc).isoformat(), warp_version=wp.__version__,
                items=ITEMS, queries=QUERIES, block_sizes=[32, 64], repeats=7,
                compiler=run(['mojo', '--version']).strip(), cpu=run(['lscpu']).strip(),
                gpu=run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']).strip(),
                samples=[], validation=[], source_hashes={}, fixture_hashes={})
    for suffix in ('boxes', 'queries'):
        path = Path(str(prefix) + '.' + suffix)
        data['fixture_hashes'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for device in ('cpu', 'cuda'):
        print('Bajo', device, flush=True)
        log = run([str(executable), device, str(prefix)])
        (work / f'bajo-{device}.log').write_text(log)
        rows = {}
        for line in log.splitlines():
            if line.startswith('SAMPLE\t'):
                _, backend, phase, ns = line.split('\t')
                rows.setdefault(phase, []).append(int(ns))
        for phase, values in rows.items():
            data['samples'].append(dict(implementation='bajo', device=device, phase=phase, ns=values))
        tree = wp.Bvh(wp.array(boxes[:, :3].copy(), dtype=wp.vec3, device=device),
                      wp.array(boxes[:, 3:].copy(), dtype=wp.vec3, device=device),
                      constructor='sah' if device == 'cpu' else 'lbvh', leaf_size=1)
        query_params = wp.array(params, dtype=wp.float32, device=device)
        for width in (32, 64):
            for kind in (0, 1):
                phase = f'query-{kind}-{width}'
                output = wp.empty(QUERIES * ITEMS, dtype=wp.float32, device=device)
                kernel = make_kernel(wp, kind)
                def launch():
                    output.zero_()
                    wp.launch_tiled(kernel, dim=[QUERIES], inputs=[tree.id, query_params, output],
                                    block_dim=width, device=device)
                values = timed(wp, device, launch)
                actual = output.numpy().reshape(QUERIES, ITEMS)
                bajo = np.fromfile(f'{prefix}-bajo-{device}-{phase}.sets', dtype='<f4').reshape(QUERIES, ITEMS)
                expected = exhaustive_sets(boxes, params, kind, False)
                # Tiled Warp ray API has no cutoff: verify the infinite ray
                # independently, rather than relying on the scalar 5.0 cutoff.
                if kind == 1:
                    p = params.reshape(QUERIES, 7)[:, :3].astype(np.float64)[:, None]
                    d = params.reshape(QUERIES, 7)[:, 3:6].astype(np.float64)[:, None]
                    t0 = (boxes[None, :, :3].astype(np.float64) - p) / d
                    t1 = (boxes[None, :, 3:].astype(np.float64) - p) / d
                    expected = np.maximum(np.minimum(t0, t1).max(axis=2), 0) <= np.maximum(t0, t1).min(axis=2)
                for name, result in (('Warp', actual), ('Bajo', bajo)):
                    if not np.array_equal(result, expected.astype(np.float32)):
                        raise RuntimeError(f'{name} {device}/{phase}: exhaustive count matrix mismatch')
                data['validation'].append(dict(device=device, phase=phase, matched=True,
                    independent=True, matches=int(actual.sum()), empty_queries=int((actual.sum(axis=1) == 0).sum())))
                data['samples'].append(dict(implementation='warp', device=device, phase=phase, ns=values))
                print(device, phase, 'validated', flush=True)
    for path in [*(ROOT / 'bajo').rglob('*.mojo'), *(ROOT / 'bench/warp').glob('*.mojo'),
                 Path(__file__), ROOT / 'bench/warp/compare_spatial.py']:
        data['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = ROOT / 'bench/results/warp/block.json'
    destination.write_text(json.dumps(data, indent=2) + '\n')
    def ratio(phase, device):
        values = {r['implementation']: statistics.median(r['ns']) for r in data['samples']
                  if r['phase'] == phase and r['device'] == device}
        return values['warp'] / values['bajo']
    lines = ['# Cooperative bounds: Bajo vs Warp', '',
             f'{data["date"][:10]} · Warp {wp.__version__} · 256 boxes, 512 queries, blocks of 32/64, seven warmed samples.', '',
             'Ratios are Warp/Bajo; above 1 favors Bajo. Both CPU paths are serial fallbacks. GPU Bajo scans preorder leaves; Warp cooperatively traverses its LBVH.', '',
             '| Operation | CPU ratio | GPU ratio |', '|---|---:|---:|']
    for phase in rows:
        _, kind, width = phase.split('-')
        lines.append(f'| {"AABB" if kind == "0" else "Ray"} / {width} lanes | {ratio(phase, "cpu"):.2f} | {ratio(phase, "cuda"):.2f} |')
    lines += ['', 'All eight CPU/GPU count matrices match Warp and exhaustive reference exactly, including empty queries and partial batches. GPU timings include a separate output-zero operation, collective query launch, output writes and synchronization; input upload/download are excluded. Warp uses atomic output increments; Bajo writes each unique emitted item directly. Ray intervals are nonnegative and effectively unbounded. Scope: one small global-bounds workload; grouped cooperative queries are covered separately by Bajo regressions. [Raw samples and hashes](block.json).', '',
              'Reproduce: `pixi run python bench/warp/compare_block.py --warp-path /tmp/bajo_warp_deps`.']
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    path = ROOT / 'bajo/bvh/PERFORMANCE.md'
    matrix = path.read_text().splitlines()
    for i, line in enumerate(matrix):
        if line.startswith('| Cooperative queries |'):
            ranges = [[ratio(phase, device) for phase in rows] for device in ('cpu', 'cuda')]
            matrix[i] = (f'| Cooperative queries | {min(ranges[0]):.2f}–{max(ranges[0]):.2f} | '
                         f'{min(ranges[1]):.2f}–{max(ranges[1]):.2f} | 256 boxes / 512 global queries; 32/64 lanes, complete sets + exhaustive reference |')
    path.write_text('\n'.join(matrix) + '\n')
    print(destination.with_suffix('.md').read_text())


if __name__ == '__main__':
    main()
