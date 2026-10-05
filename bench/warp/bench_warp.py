"""Warp side of the matched CPU/GPU geometry and GPU sorting comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--warp-path", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
args = parser.parse_args()
if args.warp_path:
    # Use the runner's NumPy ABI, even if the isolated Warp installation also
    # contains NumPy for another Python version.
    sys.path.insert(0, str(args.warp_path))
import warp as wp
import warp.utils

wp.config.kernel_cache_dir = "/tmp/bajo_warp_kernel_cache"


@wp.kernel
def trace(mesh: wp.uint64, rays: wp.array2d(dtype=wp.float32), hits: wp.array(dtype=wp.float32)):
    i = wp.tid()
    origin = wp.vec3(rays[i, 0], rays[i, 1], rays[i, 2])
    direction = wp.vec3(rays[i, 3], rays[i, 4], rays[i, 5])
    query = wp.mesh_query_ray(mesh, origin, direction, float(3.402823466e38))
    distance = float(3.402823466e38)
    if query.result:
        distance = query.t
    hits[i] = distance


@wp.kernel
def any_hit(mesh: wp.uint64, rays: wp.array2d(dtype=wp.float32), hits: wp.array(dtype=wp.float32)):
    i = wp.tid()
    origin = wp.vec3(rays[i, 0], rays[i, 1], rays[i, 2])
    direction = wp.vec3(rays[i, 3], rays[i, 4], rays[i, 5])
    hits[i] = wp.where(wp.mesh_query_ray_anyhit(mesh, origin, direction, float(3.402823466e38)),
                       float(1.0), float(0.0))


def timed(device, operation, repeats=11):
    samples = []
    for repeat in range(repeats + 2):
        wp.synchronize_device(device)
        start = time.perf_counter_ns()
        retained = operation()
        wp.synchronize_device(device)
        elapsed = time.perf_counter_ns() - start
        if repeat >= 2:
            samples.append(elapsed)
        # Keep constructed objects alive through synchronization and timing.
        del retained
    return samples


def benchmark_mesh(prefix, device, constructor, leaf_size):
    points = np.fromfile(str(prefix) + ".vertices", dtype="<f4").reshape(-1, 3)
    ray_data = np.fromfile(str(prefix) + ".rays", dtype="<f4").reshape(-1, 6)
    vertices = wp.array(points, dtype=wp.vec3, device=device)
    indices = wp.array(np.arange(len(points), dtype=np.int32), device=device)
    rays = wp.array(ray_data, dtype=wp.float32, device=device)
    hits = wp.empty(len(ray_data), dtype=wp.float32, device=device)
    def build():
        return wp.Mesh(vertices, indices, bvh_constructor=constructor, bvh_leaf_size=leaf_size)
    mesh = build()
    wp.launch(trace, dim=len(ray_data), inputs=[mesh.id, rays, hits], device=device)
    wp.synchronize_device(device)
    label = f"{device}-{constructor}-l{leaf_size}"
    result = dict(implementation="warp", scene=prefix.name, label=label,
                  triangles=len(points) // 3, rays=len(ray_data),
                  build_ns=timed(device, build),
                  trace_ns=timed(device, lambda: wp.launch(
                      trace, dim=len(ray_data), inputs=[mesh.id, rays, hits], device=device)))
    hits.numpy().astype("<f4").tofile(str(prefix) + "-warp-" + label + ".hits")
    wp.launch(any_hit, dim=len(ray_data), inputs=[mesh.id, rays, hits], device=device)
    wp.synchronize_device(device)
    result["any_ns"] = timed(device, lambda: wp.launch(
        any_hit, dim=len(ray_data), inputs=[mesh.id, rays, hits], device=device))
    hits.numpy().astype("<f4").tofile(str(prefix) + "-warp-" + label + ".any")
    return result


def sort_keys(size, pattern):
    indices = np.arange(size, dtype=np.uint32)
    if pattern == 1:
        return ((indices * np.uint32(37)) % np.uint32(257)).astype(np.int32)
    keys = (indices * np.uint32(2654435761)) ^ (indices >> np.uint32(3))
    mask = 0x00FFFFFF if pattern == 2 else 0x7FFFFFFF
    return (keys & np.uint32(mask)).astype(np.int32)


def benchmark_sort(size, pattern):
    device = "cuda:0"
    original = sort_keys(size, pattern)
    original_values = np.arange(size, dtype=np.int32)
    pristine = wp.array(original, device=device)
    ids = wp.array(original_values, device=device)
    keys = wp.empty(2 * size, dtype=wp.int32, device=device)
    values = wp.empty(2 * size, dtype=wp.int32, device=device)
    samples = []
    for repeat in range(14):
        wp.copy(keys, pristine, count=size)
        wp.copy(values, ids, count=size)
        wp.synchronize_device(device)
        start = time.perf_counter_ns()
        wp.utils.radix_sort_pairs(keys, values, size)
        wp.synchronize_device(device)
        elapsed = time.perf_counter_ns() - start
        if repeat >= 3:
            samples.append(elapsed)
    order = np.argsort(original, kind="stable")
    np.testing.assert_array_equal(keys.numpy()[:size], original[order])
    np.testing.assert_array_equal(values.numpy()[:size], original_values[order])
    return dict(implementation="warp", label="radix", size=size,
                pattern=pattern, sort_ns=samples, validated=True)


def main():
    wp.init()
    output = dict(warp_version=wp.__version__, meshes=[], sorts=[])
    for prefix in args.fixtures:
        for device, constructors in [("cpu", ["sah", "median", "cubql"]),
                                     ("cuda:0", ["lbvh", "sah", "cubql"])]:
            for constructor in constructors:
                for leaf_size in (1, 4):
                    result = benchmark_mesh(prefix, device, constructor, leaf_size)
                    output["meshes"].append(result)
                    print(prefix.name, result["label"],
                          "build ms", np.median(result["build_ns"]) / 1e6,
                          "trace ms", np.median(result["trace_ns"]) / 1e6, flush=True)
                    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for size in (65536, 1048576):
        for pattern in range(3):
            result = benchmark_sort(size, pattern)
            output["sorts"].append(result)
            print("sort", size, pattern, np.median(result["sort_ns"]) / 1e6, flush=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
