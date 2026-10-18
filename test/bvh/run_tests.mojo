"""Run the bvh Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.bvh.cpu.test_blas_assembly as cpu_test_blas_assembly
import test.bvh.cpu.test_bound_bvh as cpu_test_bound_bvh
import test.bvh.cpu.test_build_contract as cpu_test_build_contract
import test.bvh.cpu.test_gpu_triangle_math as cpu_test_gpu_triangle_math
import test.bvh.cpu.test_spatial as cpu_test_spatial
import test.bvh.cpu.test_spatial_batch as cpu_test_spatial_batch
import test.bvh.cpu.test_spatial_construction as cpu_test_spatial_construction
import test.bvh.cpu.test_spatial_distance as cpu_test_spatial_distance
import test.bvh.cpu.test_spatial_mesh as cpu_test_spatial_mesh
import test.bvh.cpu.test_spatial_subnormal_bounds as cpu_test_spatial_subnormal_bounds
import test.bvh.cpu.test_spatial_winding as cpu_test_spatial_winding
import test.bvh.cpu.test_tlas as cpu_test_tlas
import test.bvh.gpu.test_blas_tlas as gpu_test_blas_tlas
import test.bvh.gpu.test_bounds_bvh as gpu_test_bounds_bvh
import test.bvh.gpu.test_build_finalize as gpu_test_build_finalize
import test.bvh.gpu.test_bvh2_closest_tasks as gpu_test_bvh2_closest_tasks
import test.bvh.gpu.test_cwbvh8_node_tasks as gpu_test_cwbvh8_node_tasks
import test.bvh.gpu.test_hit_store as gpu_test_hit_store
import test.bvh.gpu.test_launch_contracts as gpu_test_launch_contracts
import test.bvh.gpu.test_ray_ordering as gpu_test_ray_ordering
import test.bvh.gpu.test_segmented_empty as gpu_test_segmented_empty
import test.bvh.gpu.test_spatial as gpu_test_spatial
import test.bvh.gpu.test_spatial_block as gpu_test_spatial_block
import test.bvh.gpu.test_spatial_bounds_parallel_updates as gpu_test_spatial_bounds_parallel_updates
import test.bvh.gpu.test_spatial_bounds_precision as gpu_test_spatial_bounds_precision
import test.bvh.gpu.test_spatial_device_updates as gpu_test_spatial_device_updates
import test.bvh.gpu.test_spatial_furthest_precision as gpu_test_spatial_furthest_precision
import test.bvh.gpu.test_spatial_gpu_geometry as gpu_test_spatial_gpu_geometry
import test.bvh.gpu.test_spatial_mesh as gpu_test_spatial_mesh
import test.bvh.gpu.test_spatial_nearest_precision as gpu_test_spatial_nearest_precision
import test.bvh.gpu.test_spatial_normal_sign_filter as gpu_test_spatial_normal_sign_filter
import test.bvh.gpu.test_spatial_parallel_refit as gpu_test_spatial_parallel_refit
import test.bvh.gpu.test_spatial_ray_precision as gpu_test_spatial_ray_precision
import test.bvh.gpu.test_spatial_sphere_precision as gpu_test_spatial_sphere_precision
import test.bvh.gpu.test_spatial_subnormal_bounds as gpu_test_spatial_subnormal_bounds
import test.bvh.gpu.test_spatial_velocity_updates as gpu_test_spatial_velocity_updates
import test.bvh.gpu.test_spatial_winding_cache as gpu_test_spatial_winding_cache
import test.bvh.gpu.test_tlas as gpu_test_tlas
import test.bvh.gpu.test_tlas_visibility as gpu_test_tlas_visibility
import test.bvh.gpu.test_triangle_shadow as gpu_test_triangle_shadow


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["cpu_test_blas_assembly"]():
        tests.run(
            "test/bvh/cpu/test_blas_assembly.mojo",
            cpu_test_blas_assembly.run_tests[True],
        )
    comptime if module_selected["cpu_test_bound_bvh"]():
        tests.run(
            "test/bvh/cpu/test_bound_bvh.mojo",
            cpu_test_bound_bvh.run_tests[True],
        )
    comptime if module_selected["cpu_test_build_contract"]():
        tests.run(
            "test/bvh/cpu/test_build_contract.mojo",
            cpu_test_build_contract.run_tests[True],
        )
    comptime if module_selected["cpu_test_gpu_triangle_math"]():
        tests.run(
            "test/bvh/cpu/test_gpu_triangle_math.mojo",
            cpu_test_gpu_triangle_math.run_tests[True],
        )
    comptime if module_selected["cpu_test_spatial"]():
        tests.run(
            "test/bvh/cpu/test_spatial.mojo", cpu_test_spatial.run_tests[True]
        )
    comptime if module_selected["cpu_test_spatial_batch"]():
        tests.run(
            "test/bvh/cpu/test_spatial_batch.mojo",
            cpu_test_spatial_batch.run_tests[True],
        )
    comptime if module_selected["cpu_test_spatial_construction"]():
        tests.run(
            "test/bvh/cpu/test_spatial_construction.mojo",
            cpu_test_spatial_construction.run_tests[True],
        )
    comptime if module_selected["cpu_test_spatial_distance"]():
        tests.run(
            "test/bvh/cpu/test_spatial_distance.mojo",
            cpu_test_spatial_distance.run_tests[True],
        )
    comptime if module_selected["cpu_test_spatial_mesh"]():
        tests.run(
            "test/bvh/cpu/test_spatial_mesh.mojo",
            cpu_test_spatial_mesh.run_tests[True],
        )
    comptime if module_selected["cpu_test_spatial_subnormal_bounds"]():
        tests.run(
            "test/bvh/cpu/test_spatial_subnormal_bounds.mojo",
            cpu_test_spatial_subnormal_bounds.run_tests[True],
        )
    comptime if module_selected["cpu_test_spatial_winding"]():
        tests.run(
            "test/bvh/cpu/test_spatial_winding.mojo",
            cpu_test_spatial_winding.run_tests[True],
        )
    comptime if module_selected["cpu_test_tlas"]():
        tests.run("test/bvh/cpu/test_tlas.mojo", cpu_test_tlas.run_tests[True])
    comptime if module_selected["gpu_test_blas_tlas"]():
        tests.run(
            "test/bvh/gpu/test_blas_tlas.mojo",
            gpu_test_blas_tlas.run_tests[True],
        )
    comptime if module_selected["gpu_test_bounds_bvh"]():
        tests.run(
            "test/bvh/gpu/test_bounds_bvh.mojo",
            gpu_test_bounds_bvh.run_tests[True],
        )
    comptime if module_selected["gpu_test_build_finalize"]():
        tests.run(
            "test/bvh/gpu/test_build_finalize.mojo",
            gpu_test_build_finalize.run_tests[True],
        )
    comptime if module_selected["gpu_test_bvh2_closest_tasks"]():
        tests.run(
            "test/bvh/gpu/test_bvh2_closest_tasks.mojo",
            gpu_test_bvh2_closest_tasks.run_tests[True],
        )
    comptime if module_selected["gpu_test_cwbvh8_node_tasks"]():
        tests.run(
            "test/bvh/gpu/test_cwbvh8_node_tasks.mojo",
            gpu_test_cwbvh8_node_tasks.run_tests[True],
        )
    comptime if module_selected["gpu_test_hit_store"]():
        tests.run(
            "test/bvh/gpu/test_hit_store.mojo",
            gpu_test_hit_store.run_tests[True],
        )
    comptime if module_selected["gpu_test_launch_contracts"]():
        tests.run(
            "test/bvh/gpu/test_launch_contracts.mojo",
            gpu_test_launch_contracts.run_tests[True],
        )
    comptime if module_selected["gpu_test_ray_ordering"]():
        tests.run(
            "test/bvh/gpu/test_ray_ordering.mojo",
            gpu_test_ray_ordering.run_tests[True],
        )
    comptime if module_selected["gpu_test_segmented_empty"]():
        tests.run(
            "test/bvh/gpu/test_segmented_empty.mojo",
            gpu_test_segmented_empty.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial"]():
        tests.run(
            "test/bvh/gpu/test_spatial.mojo", gpu_test_spatial.run_tests[True]
        )
    comptime if module_selected["gpu_test_spatial_block"]():
        tests.run(
            "test/bvh/gpu/test_spatial_block.mojo",
            gpu_test_spatial_block.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_bounds_parallel_updates"]():
        tests.run(
            "test/bvh/gpu/test_spatial_bounds_parallel_updates.mojo",
            gpu_test_spatial_bounds_parallel_updates.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_bounds_precision"]():
        tests.run(
            "test/bvh/gpu/test_spatial_bounds_precision.mojo",
            gpu_test_spatial_bounds_precision.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_device_updates"]():
        tests.run(
            "test/bvh/gpu/test_spatial_device_updates.mojo",
            gpu_test_spatial_device_updates.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_furthest_precision"]():
        tests.run(
            "test/bvh/gpu/test_spatial_furthest_precision.mojo",
            gpu_test_spatial_furthest_precision.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_gpu_geometry"]():
        tests.run(
            "test/bvh/gpu/test_spatial_gpu_geometry.mojo",
            gpu_test_spatial_gpu_geometry.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_mesh"]():
        tests.run(
            "test/bvh/gpu/test_spatial_mesh.mojo",
            gpu_test_spatial_mesh.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_nearest_precision"]():
        tests.run(
            "test/bvh/gpu/test_spatial_nearest_precision.mojo",
            gpu_test_spatial_nearest_precision.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_normal_sign_filter"]():
        tests.run(
            "test/bvh/gpu/test_spatial_normal_sign_filter.mojo",
            gpu_test_spatial_normal_sign_filter.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_parallel_refit"]():
        tests.run(
            "test/bvh/gpu/test_spatial_parallel_refit.mojo",
            gpu_test_spatial_parallel_refit.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_ray_precision"]():
        tests.run(
            "test/bvh/gpu/test_spatial_ray_precision.mojo",
            gpu_test_spatial_ray_precision.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_sphere_precision"]():
        tests.run(
            "test/bvh/gpu/test_spatial_sphere_precision.mojo",
            gpu_test_spatial_sphere_precision.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_subnormal_bounds"]():
        tests.run(
            "test/bvh/gpu/test_spatial_subnormal_bounds.mojo",
            gpu_test_spatial_subnormal_bounds.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_velocity_updates"]():
        tests.run(
            "test/bvh/gpu/test_spatial_velocity_updates.mojo",
            gpu_test_spatial_velocity_updates.run_tests[True],
        )
    comptime if module_selected["gpu_test_spatial_winding_cache"]():
        tests.run(
            "test/bvh/gpu/test_spatial_winding_cache.mojo",
            gpu_test_spatial_winding_cache.run_tests[True],
        )
    comptime if module_selected["gpu_test_tlas"]():
        tests.run("test/bvh/gpu/test_tlas.mojo", gpu_test_tlas.run_tests[True])
    comptime if module_selected["gpu_test_tlas_visibility"]():
        tests.run(
            "test/bvh/gpu/test_tlas_visibility.mojo",
            gpu_test_tlas_visibility.run_tests[True],
        )
    comptime if module_selected["gpu_test_triangle_shadow"]():
        tests.run(
            "test/bvh/gpu/test_triangle_shadow.mojo",
            gpu_test_triangle_shadow.run_tests[True],
        )
    tests^.finish()
