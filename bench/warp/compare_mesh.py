"""Forward mesh conformance/performance against Warp; VJPs are excluded."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np  # Load the runner ABI before the isolated Warp installation.
from compare_spatial import ROOT, run, timed

ITEMS, QUERIES = 256, 256
KINDS = ('closest', 'furthest', 'normal sign', 'parity sign', 'winding sign',
         'closest ray', 'ray count', 'any-hit ray', 'sphere', 'AABB')
CASES = [(k, False) for k in range(10)] + [(k, True) for k in (5, 6, 7, 9)]


def make_kernel(wp, kind, grouped):
    @wp.kernel
    def query(mesh: wp.uint64, params: wp.array(dtype=wp.float32),
              output: wp.array(dtype=wp.float32), width: int):
        q = wp.tid()
        for j in range(width):
            output[q * width + j] = float(0.0)
        p = wp.vec3(params[q * 7], params[q * 7 + 1], params[q * 7 + 2])
        d = wp.vec3(params[q * 7 + 3], params[q * 7 + 4], params[q * 7 + 5])
        root = int(-1)
        if wp.static(grouped):
            root = wp.mesh_get_group_root(mesh, q % 4)
        if wp.static(kind < 5):
            if wp.static(kind == 0):
                hit = wp.mesh_query_point_no_sign(mesh, p, float(1000.0))
            elif wp.static(kind == 1):
                hit = wp.mesh_query_furthest_point_no_sign(mesh, p, float(0.0))
            elif wp.static(kind == 2):
                hit = wp.mesh_query_point_sign_normal(mesh, p, float(1000.0))
            elif wp.static(kind == 3):
                hit = wp.mesh_query_point_sign_parity(mesh, p, float(1000.0))
            else:
                hit = wp.mesh_query_point_sign_winding_number(mesh, p, float(1000.0))
            if hit.result:
                pos = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
                vel = wp.mesh_eval_velocity(mesh, hit.face, hit.u, hit.v)
                output[q * width] = float(1.0)
                output[q * width + 1] = pos[0]
                output[q * width + 2] = pos[1]
                output[q * width + 3] = pos[2]
                output[q * width + 4] = wp.length(p - pos)
                output[q * width + 5] = vel[0]
                output[q * width + 6] = vel[1]
                output[q * width + 7] = vel[2]
                if wp.static(kind >= 2):
                    output[q * width + 8] = hit.sign
                output[q * width + 9] = float(hit.face)
        elif wp.static(kind == 5):
            ray = wp.mesh_query_ray(mesh, p, d, float(1000.0), root)
            if ray.result:
                pos = wp.mesh_eval_position(mesh, ray.face, ray.u, ray.v)
                output[q * width] = float(1.0)
                output[q * width + 1] = ray.t
                output[q * width + 2] = ray.normal[0]
                output[q * width + 3] = ray.normal[1]
                output[q * width + 4] = ray.normal[2]
                output[q * width + 5] = pos[0]
                output[q * width + 6] = pos[1]
                output[q * width + 7] = pos[2]
                output[q * width + 8] = wp.where(ray.sign > 0.0, float(1.0), float(0.0))
                output[q * width + 9] = float(ray.face)
        elif wp.static(kind == 6):
            output[q * width] = float(wp.mesh_query_ray_count_intersections(mesh, p, d, root))
        elif wp.static(kind == 7):
            output[q * width] = wp.where(wp.mesh_query_ray_anyhit(mesh, p, d, float(1000.0), root), float(1.0), float(0.0))
        elif wp.static(kind == 8):
            sphere = wp.mesh_query_sphere(mesh, p, params[q * 7 + 6])
            face = int(0)
            while wp.mesh_query_next(sphere, face):
                output[q * width + face] += float(1.0)
        else:
            r = wp.vec3(params[q * 7 + 6])
            box = wp.bvh_query_aabb(wp.mesh_get_bvh(mesh), p - r, p + r, root)
            face = int(0)
            while wp.bvh_query_next(box, face):
                output[q * width + face] += float(1.0)
    return query


def exhaustive(points, indices, params):
    """Float64 exhaustive projection/rays plus analytic disjoint tetrahedron signs."""
    tri = points.astype(np.float64)[indices.reshape(-1, 3)]
    p = params[:, :3].astype(np.float64)[:, None, :]
    a, b, c = (tri[None, :, i, :] for i in range(3))
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    n2 = (normal * normal).sum(axis=2)
    plane = p - normal * ((p - a) * normal).sum(axis=2)[:, :, None] / n2[:, :, None]
    dot00, dot01, dot11 = (ab * ab).sum(axis=2), (ab * ac).sum(axis=2), (ac * ac).sum(axis=2)
    dot20, dot21 = ((plane - a) * ab).sum(axis=2), ((plane - a) * ac).sum(axis=2)
    den = dot00 * dot11 - dot01 ** 2
    u = (dot11 * dot20 - dot01 * dot21) / den
    v = (dot00 * dot21 - dot01 * dot20) / den
    candidates = [plane]
    distances = [np.where((u >= 0) & (v >= 0) & (u + v <= 1), ((p - plane) ** 2).sum(axis=2), np.inf)]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.clip(((p - start) * edge).sum(axis=2) / (edge * edge).sum(axis=2), 0, 1)
        pos = start + t[:, :, None] * edge
        candidates.append(pos)
        distances.append(((p - pos) ** 2).sum(axis=2))
    distance2 = np.stack(distances).min(axis=0)
    chosen = np.stack(distances).argmin(axis=0)
    positions = np.stack(candidates)[chosen, np.arange(QUERIES)[:, None], np.arange(ITEMS)[None, :]]
    face = distance2.argmin(axis=1)
    nearest = positions[np.arange(QUERIES), face]
    far_distance2 = ((params[:, None, :3].astype(np.float64) - points[None].astype(np.float64)) ** 2).sum(axis=2)
    far = points[far_distance2.argmax(axis=1)]
    local = p - points[::4].astype(np.float64)[None]
    edges = points[1::4].astype(np.float64)[:, 0] - points[::4].astype(np.float64)[:, 0]
    inside = ((local > 0).all(axis=2) & ((local / edges[None, :, None]).sum(axis=2) < 1)).any(axis=1)
    d = params[:, 3:6].astype(np.float64)[:, None, :]
    h = np.cross(d, ac)
    determinant = (ab * h).sum(axis=2)
    inv = 1 / determinant
    s = p - a
    ru = (s * h).sum(axis=2) * inv
    cross = np.cross(s, ab)
    rv = (d * cross).sum(axis=2) * inv
    rt = (ac * cross).sum(axis=2) * inv
    ray = (ru >= 0) & (rv >= 0) & (ru + rv <= 1) & (rt >= 0)
    return dict(nearest=nearest, near_distance=np.sqrt(distance2.min(axis=1)), far=far,
                far_distance=np.sqrt(far_distance2.max(axis=1)), sign=np.where(inside, -1, 1),
                sphere=distance2 <= params[:, 6, None].astype(np.float64) ** 2,
                ray=ray, rt=rt, normals=normal[0] / np.sqrt(n2[0])[:, None], tri=tri)


def reference_rows(reference, points, params, kind, grouped):
    width = ITEMS if kind >= 8 else 10
    out = np.zeros((QUERIES, width), dtype=np.float64)
    mask = np.ones((QUERIES, ITEMS), dtype=bool)
    if grouped:
        mask &= np.arange(QUERIES)[:, None] % 4 == (np.arange(ITEMS)[None, :] // 4) % 4
    if kind < 5:
        out[:, 0] = 1
        pos = reference['far'] if kind == 1 else reference['nearest']
        out[:, 1:4] = pos
        out[:, 4] = reference['far_distance'] if kind == 1 else reference['near_distance']
        out[:, 5:8] = pos * [2, 3, 4] + [1, -1, .5]
        if kind >= 2:
            out[:, 8] = reference['sign']
    elif kind in (5, 6, 7):
        hits = reference['ray'] & mask
        if kind == 6:
            out[:, 0] = hits.sum(axis=1)
        elif kind == 7:
            out[:, 0] = hits.any(axis=1)
        else:
            ts = np.where(hits, reference['rt'], np.inf)
            face = ts.argmin(axis=1)
            hit = hits.any(axis=1)
            t = ts[np.arange(QUERIES), face][hit]
            out[hit, 0] = 1
            out[hit, 1] = t
            normals = reference['normals'][face[hit]]
            out[hit, 2:5] = normals
            out[hit, 5:8] = params[hit, :3] + t[:, None] * params[hit, 3:6]
            out[hit, 8] = (normals * params[hit, 3:6]).sum(axis=1) < 0
            out[hit, 9] = face[hit]
    elif kind == 8:
        out = reference['sphere'].astype(np.float64)
    else:
        lo, hi = reference['tri'].min(axis=1), reference['tri'].max(axis=1)
        qlo = (params[:, :3] - params[:, 6, None]).astype(np.float64)
        qhi = (params[:, :3] + params[:, 6, None]).astype(np.float64)
        out = (((qlo[:, None] <= hi[None]) & (qhi[:, None] >= lo[None])).all(axis=2) & mask).astype(np.float64)
    return out


def check(actual, expected, kind, label):
    columns = slice(None, 9) if kind < 5 else slice(None)
    # Different faces can describe the same shared edge/vertex. Compare the
    # reconstructed position/distance/velocity, rather than arbitrary tie IDs.
    left, right = actual[:, columns], expected[:, columns]
    if not np.isfinite(left).all() or not np.allclose(left, right, atol=5e-6, rtol=5e-6):
        bad = np.argwhere(~np.isclose(left, right, atol=5e-6, rtol=5e-6))
        raise RuntimeError(f'{label}: {len(bad)} fields differ; first {bad[:5].tolist()}; '
                           f'actual {left[tuple(bad[0])] if len(bad) else "nonfinite"}, '
                           f'expected {right[tuple(bad[0])] if len(bad) else "nonfinite"}')
    # Discrete answers and candidate multiplicities must match exactly.
    discrete = slice(None) if kind >= 8 else ([0, 8] if 2 <= kind <= 5 else [0])
    if not np.array_equal(actual[:, discrete], expected[:, discrete]):
        raise RuntimeError(f'{label}: discrete results differ')


def refresh_matrix(data):
    def ratio(phase, device):
        values = {r['implementation']: statistics.median(r['ns']) for r in data['samples']
                  if r['device'] == device and r['phase'] == phase}
        return values['warp'] / values['bajo']
    path = ROOT / 'bajo/bvh/PERFORMANCE.md'
    lines = path.read_text().splitlines()
    mapping = {'Exact mesh sphere query': [8], 'Closest/furthest mesh point': [0, 1],
               'Mesh sign classification': [2, 3, 4], 'Ray intersection count': [6],
               'Group-restricted mesh queries': [5, 6, 7, 9]}
    for i, line in enumerate(lines):
        parts = line.split('|')
        if len(parts) < 6:
            continue
        name = parts[1].strip()
        if name in mapping:
            grouped = int(name == 'Group-restricted mesh queries')
            for column, device in ((2, 'cpu'), (3, 'cuda')):
                ratios = [ratio(f'query-{k}-{grouped}', device) for k in mapping[name]]
                parts[column] = f' {min(ratios):.2f}–{max(ratios):.2f} ' if len(ratios) > 1 else f' {ratios[0]:.2f} '
            parts[4] = ' 256 faces / 256 queries; matched Warp + exhaustive/analytic reference '
        elif name == 'Mesh deformation refit':
            parts[2], parts[3] = f' {ratio("refit", "cpu"):.2f} ', f' {ratio("refit", "cuda"):.2f} '
            parts[4] = ' Warm unchanged positions; GPU resident inputs, four groups; Warp winding support disabled '
        else:
            continue
        lines[i] = '|'.join(parts)
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warp-path', type=Path)
    parser.add_argument('--gpu-arch', default='sm_120')
    parser.add_argument('--work-dir', type=Path, default=Path('/tmp/bajo_warp_mesh'))
    args = parser.parse_args()
    if args.warp_path:
        sys.path.insert(0, str(args.warp_path))
    import warp as wp
    wp.config.kernel_cache_dir = '/tmp/bajo_warp_mesh_cache'
    wp.init()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    executable = work / 'mesh'
    run(['mojo', 'build', '-I', str(ROOT), '--target-accelerator', args.gpu_arch,
         str(ROOT / 'bench/warp/bench_mesh.mojo'), '-o', str(executable)])
    prefix = work / 'fixture'
    run([str(executable), 'fixtures', str(prefix)])
    fixture = {name: np.fromfile(str(prefix) + '.' + name, dtype='<f4')
               for name in ('points', 'indices', 'velocities', 'queries')}
    points = fixture['points'].reshape(-1, 3)
    indices = fixture['indices'].astype(np.int32)
    params = fixture['queries'].reshape(QUERIES, 7)
    groups = (np.arange(ITEMS, dtype=np.int32) // 4) % 4
    reference = exhaustive(points, indices, params)
    data = dict(date=datetime.now(timezone.utc).isoformat(), warp_version=wp.__version__,
                faces=ITEMS, queries=QUERIES, groups=4, repeats=7, samples=[], validation=[],
                compiler=run(['mojo', '--version']).strip(), cpu=run(['lscpu']).strip(),
                gpu=run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']).strip(),
                fixture_hashes={}, source_hashes={}, excluded_failures=[], tolerance=dict(atol=5e-6, rtol=5e-6))
    for name in fixture:
        path = Path(str(prefix) + '.' + name)
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
        for phase, samples in rows.items():
            data['samples'].append(dict(implementation='bajo', device=device, phase=phase, ns=samples))
        tree = wp.Mesh(wp.array(points, dtype=wp.vec3, device=device),
                       wp.array(indices, dtype=wp.int32, device=device),
                       velocities=wp.array(fixture['velocities'].reshape(-1, 3), dtype=wp.vec3, device=device),
                       groups=wp.array(groups, dtype=wp.int32, device=device), support_winding_number=False,
                       bvh_constructor='sah' if device == 'cpu' else 'lbvh', bvh_leaf_size=1)
        data['samples'].append(dict(implementation='warp', device=device, phase='refit',
                                    ns=timed(wp, device, tree.refit)))
        winding_tree = wp.Mesh(wp.array(points, dtype=wp.vec3, device=device),
                               wp.array(indices, dtype=wp.int32, device=device),
                               velocities=wp.array(fixture['velocities'].reshape(-1, 3), dtype=wp.vec3, device=device),
                               support_winding_number=True,
                               bvh_constructor='sah' if device == 'cpu' else 'lbvh', bvh_leaf_size=1)
        query_params = wp.array(params.ravel(), dtype=wp.float32, device=device)
        if device == 'cpu':
            grouped_winding = wp.Mesh(wp.array(points, dtype=wp.vec3, device=device),
                wp.array(indices, dtype=wp.int32, device=device),
                groups=wp.array(groups, dtype=wp.int32, device=device), support_winding_number=True,
                bvh_constructor='sah', bvh_leaf_size=1)
            grouped_winding.refit()
            failure_output = wp.empty(QUERIES * 10, dtype=wp.float32, device=device)
            wp.launch(make_kernel(wp, 4, False), dim=QUERIES,
                inputs=[grouped_winding.id, query_params, failure_output, 10], device=device)
            failure_signs = failure_output.numpy().reshape(QUERIES, 10)[:, 8]
            mismatches = np.flatnonzero(failure_signs != reference['sign'])
            if len(mismatches):
                data['excluded_failures'].append(dict(implementation='warp', device=device,
                    configuration='grouped winding support + refit', timed=False,
                    sign_mismatches=int(len(mismatches)), query_indices=mismatches.tolist(),
                    actual_signs=failure_signs[mismatches].tolist(),
                    expected_signs=reference['sign'][mismatches].tolist()))
                print('Excluded grouped Warp CPU winding:', len(mismatches), 'sign mismatches', flush=True)
        for kind, grouped in CASES:
            phase = f'query-{kind}-{int(grouped)}'
            width = ITEMS if kind >= 8 else 10
            output = wp.empty(QUERIES * width, dtype=wp.float32, device=device)
            kernel = make_kernel(wp, kind, grouped)
            values = timed(wp, device, lambda: wp.launch(kernel, dim=QUERIES,
                           inputs=[winding_tree.id if kind == 4 else tree.id, query_params, output, width], device=device))
            warp = output.numpy().reshape(QUERIES, width)
            bajo = np.fromfile(f'{prefix}-bajo-{device}-{phase}.results', dtype='<f4').reshape(QUERIES, width)
            expected = reference_rows(reference, points, params, kind, grouped)
            for name, result in (('Warp', warp), ('Bajo', bajo)):
                check(result, expected, kind, f'{name} {device}/{phase} reference')
            check(bajo, warp, kind, f'{device}/{phase} matched')
            data['validation'].append(dict(device=device, phase=phase, matched=True, independent=True))
            data['samples'].append(dict(implementation='warp', device=device, phase=phase, ns=values))
            print(device, KINDS[kind], 'group' if grouped else 'global', 'validated', flush=True)
    for path in [*(ROOT / 'bajo').rglob('*.mojo'), *(ROOT / 'bench/warp').glob('*.mojo'),
                 Path(__file__), ROOT / 'bench/warp/compare_spatial.py']:
        data['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = ROOT / 'bench/results/warp/mesh.json'
    destination.write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# Indexed mesh: Bajo vs Warp', '',
             f'{data["date"][:10]} · Warp {wp.__version__} · 64 disjoint outward tetrahedra, 256 faces, four groups, 256 queries, seven warmed samples.', '',
             'Ratios are Warp/Bajo; above 1 favors Bajo. CPU median/SAH and GPU median-upload/LBVH, one face per leaf.', '',
             '| Operation | CPU ratio | GPU ratio |', '|---|---:|---:|']
    for phase in rows:
        ratios = []
        for device in ('cpu', 'cuda'):
            values = {r['implementation']: statistics.median(r['ns']) for r in data['samples'] if r['phase'] == phase and r['device'] == device}
            ratios.append(values['warp'] / values['bajo'])
        label = 'Refit' if phase == 'refit' else KINDS[int(phase.split('-')[1])] + (' / group' if phase.endswith('-1') else ' / global')
        lines.append(f'| {label} | {ratios[0]:.2f} | {ratios[1]:.2f} |')
    lines += ['', 'All 28 CPU/GPU query cases match Warp and an independent exhaustive Float64 geometry / analytic tetrahedron reference. Discrete answers and overlap multiplicities match exactly; floating fields use 5e-6 absolute/relative tolerance. Shared-edge/vertex nearest and furthest ties compare reconstructed positions, distances and velocities, allowing different face IDs.', '',
              'Query timings include outputs, launch and synchronization; download and fixture upload are excluded. Refit uses unchanged warmed positions; GPU inputs are resident, Bajo validation is included and status download is excluded. The winding case uses separate ungrouped owners with Warp winding support enabled; other cases/refit use grouped owners without Warp winding support. Bajo sums exact solid angles while Warp uses its accelerated winding approximation. The raw record includes a separate correctness check of grouped Warp CPU winding; any failed configuration is excluded from timings. This small workload does not establish scaling or large-deformation performance. Cutoffs and query points avoid semantic boundaries. [Raw samples and hashes](mesh.json).', '',
              'Reproduce: `pixi run python bench/warp/compare_mesh.py --warp-path /tmp/bajo_warp_deps`.']
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    refresh_matrix(data)
    print(destination.with_suffix('.md').read_text())


if __name__ == '__main__':
    main()
