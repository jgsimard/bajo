"""Run the rt Mojo tests in one process."""

from test.runner import TestRunner, module_selected

import test.rt.cpu.test_bmw_materials as cpu_test_bmw_materials
import test.rt.cpu.test_coated_diffuse as cpu_test_coated_diffuse
import test.rt.cpu.test_coated_simd as cpu_test_coated_simd
import test.rt.cpu.test_cpu as cpu_test_cpu
import test.rt.cpu.test_dielectric as cpu_test_dielectric
import test.rt.cpu.test_opacity as cpu_test_opacity
import test.rt.cpu.test_texture_storage as cpu_test_texture_storage
import test.rt.cpu.test_coated_reflectance as cpu_test_coated_reflectance
import test.rt.cpu.test_emissive_bsdf as cpu_test_emissive_bsdf
import test.rt.cpu.test_film as cpu_test_film
import test.rt.cpu.test_filter as cpu_test_filter
import test.rt.cpu.test_spectra as cpu_test_spectra
import test.rt.cpu.test_crown as cpu_test_crown
import test.rt.cpu.test_gpu_material_layout as cpu_test_gpu_material_layout
import test.rt.cpu.test_gpu_path_continuation as cpu_test_gpu_path_continuation
import test.rt.cpu.test_instance_shading_simd as cpu_test_instance_shading_simd
import test.rt.cpu.test_packet_inputs as cpu_test_packet_inputs
import test.rt.cpu.test_packet_tiles as cpu_test_packet_tiles
import test.rt.cpu.test_sampling_simd as cpu_test_sampling_simd
import test.rt.cpu.test_scene_preparation as cpu_test_scene_preparation
import test.rt.cpu.test_scene_shapes as cpu_test_scene_shapes
import test.rt.cpu.test_wavefront_contract as cpu_test_wavefront_contract
import test.rt.gpu.test_bmw_materials_gpu as gpu_test_bmw_materials_gpu
import test.rt.gpu.test_bistro_gpu as gpu_test_bistro_gpu
import test.rt.gpu.test_opacity_gpu as gpu_test_opacity_gpu
import test.rt.gpu.test_texture_storage_gpu as gpu_test_texture_storage_gpu
import test.rt.gpu.test_emissive_bsdf_gpu as gpu_test_emissive_bsdf_gpu
import test.rt.gpu.test_film_gpu as gpu_test_film_gpu
import test.rt.gpu.test_filter_gpu as gpu_test_filter_gpu
import test.rt.gpu.test_spectra_gpu as gpu_test_spectra_gpu
import test.rt.gpu.test_coated_shading_gpu as gpu_test_coated_shading_gpu
import test.rt.gpu.test_crown_gpu as gpu_test_crown_gpu
import test.rt.gpu.test_cwbvh_call_gpu as gpu_test_cwbvh_call_gpu
import test.rt.gpu.test_delta_metal_gpu as gpu_test_delta_metal_gpu
import test.rt.gpu.test_gpu as gpu_test_gpu
import test.rt.gpu.test_primary_launch_gpu as gpu_test_primary_launch_gpu
import test.rt.gpu.test_ray_order_gpu as gpu_test_ray_order_gpu
import test.rt.gpu.test_target_capacity_gpu as gpu_test_target_capacity_gpu
import test.rt.gpu.test_wavefront_contract_gpu as gpu_test_wavefront_contract_gpu


def main() raises:
    var tests = TestRunner()
    comptime if module_selected["gpu_test_spectra_gpu"]():
        tests.run(
            "test/rt/gpu/test_spectra_gpu.mojo",
            gpu_test_spectra_gpu.run_tests[True],
        )
    comptime if module_selected["cpu_test_spectra"]():
        tests.run(
            "test/rt/cpu/test_spectra.mojo", cpu_test_spectra.run_tests[True]
        )
    comptime if module_selected["cpu_test_filter"]():
        tests.run(
            "test/rt/cpu/test_filter.mojo", cpu_test_filter.run_tests[True]
        )
    comptime if module_selected["gpu_test_filter_gpu"]():
        tests.run(
            "test/rt/gpu/test_filter_gpu.mojo",
            gpu_test_filter_gpu.run_tests[True],
        )
    comptime if module_selected["cpu_test_film"]():
        tests.run("test/rt/cpu/test_film.mojo", cpu_test_film.run_tests[True])
    comptime if module_selected["gpu_test_film_gpu"]():
        tests.run(
            "test/rt/gpu/test_film_gpu.mojo", gpu_test_film_gpu.run_tests[True]
        )
    comptime if module_selected["cpu_test_bmw_materials"]():
        tests.run(
            "test/rt/cpu/test_bmw_materials.mojo",
            cpu_test_bmw_materials.run_tests[True],
        )
    comptime if module_selected["cpu_test_coated_diffuse"]():
        tests.run(
            "test/rt/cpu/test_coated_diffuse.mojo",
            cpu_test_coated_diffuse.run_tests[True],
        )
    comptime if module_selected["cpu_test_coated_simd"]():
        tests.run(
            "test/rt/cpu/test_coated_simd.mojo",
            cpu_test_coated_simd.run_tests[True],
        )
    comptime if module_selected["cpu_test_cpu"]():
        tests.run("test/rt/cpu/test_cpu.mojo", cpu_test_cpu.run_tests[True])
    comptime if module_selected["cpu_test_dielectric"]():
        tests.run(
            "test/rt/cpu/test_dielectric.mojo",
            cpu_test_dielectric.run_tests[True],
        )
    comptime if module_selected["cpu_test_opacity"]():
        tests.run(
            "test/rt/cpu/test_opacity.mojo", cpu_test_opacity.run_tests[True]
        )
    comptime if module_selected["cpu_test_texture_storage"]():
        tests.run(
            "test/rt/cpu/test_texture_storage.mojo",
            cpu_test_texture_storage.run_tests[True],
        )
    comptime if module_selected["cpu_test_coated_reflectance"]():
        tests.run(
            "test/rt/cpu/test_coated_reflectance.mojo",
            cpu_test_coated_reflectance.run_tests[True],
        )
    comptime if module_selected["cpu_test_emissive_bsdf"]():
        tests.run(
            "test/rt/cpu/test_emissive_bsdf.mojo",
            cpu_test_emissive_bsdf.run_tests[True],
        )
    comptime if module_selected["cpu_test_crown"]():
        tests.run("test/rt/cpu/test_crown.mojo", cpu_test_crown.run_tests[True])
    comptime if module_selected["cpu_test_gpu_material_layout"]():
        tests.run(
            "test/rt/cpu/test_gpu_material_layout.mojo",
            cpu_test_gpu_material_layout.run_tests[True],
        )
    comptime if module_selected["cpu_test_gpu_path_continuation"]():
        tests.run(
            "test/rt/cpu/test_gpu_path_continuation.mojo",
            cpu_test_gpu_path_continuation.run_tests[True],
        )
    comptime if module_selected["cpu_test_instance_shading_simd"]():
        tests.run(
            "test/rt/cpu/test_instance_shading_simd.mojo",
            cpu_test_instance_shading_simd.run_tests[True],
        )
    comptime if module_selected["cpu_test_packet_inputs"]():
        tests.run(
            "test/rt/cpu/test_packet_inputs.mojo",
            cpu_test_packet_inputs.run_tests[True],
        )
    comptime if module_selected["cpu_test_packet_tiles"]():
        tests.run(
            "test/rt/cpu/test_packet_tiles.mojo",
            cpu_test_packet_tiles.run_tests[True],
        )
    comptime if module_selected["cpu_test_sampling_simd"]():
        tests.run(
            "test/rt/cpu/test_sampling_simd.mojo",
            cpu_test_sampling_simd.run_tests[True],
        )
    comptime if module_selected["cpu_test_scene_preparation"]():
        tests.run(
            "test/rt/cpu/test_scene_preparation.mojo",
            cpu_test_scene_preparation.run_tests[True],
        )
    comptime if module_selected["cpu_test_scene_shapes"]():
        tests.run(
            "test/rt/cpu/test_scene_shapes.mojo",
            cpu_test_scene_shapes.run_tests[True],
        )
    comptime if module_selected["cpu_test_wavefront_contract"]():
        tests.run(
            "test/rt/cpu/test_wavefront_contract.mojo",
            cpu_test_wavefront_contract.run_tests[True],
        )
    comptime if module_selected["gpu_test_opacity_gpu"]():
        tests.run(
            "test/rt/gpu/test_opacity_gpu.mojo",
            gpu_test_opacity_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_texture_storage_gpu"]():
        tests.run(
            "test/rt/gpu/test_texture_storage_gpu.mojo",
            gpu_test_texture_storage_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_emissive_bsdf_gpu"]():
        tests.run(
            "test/rt/gpu/test_emissive_bsdf_gpu.mojo",
            gpu_test_emissive_bsdf_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_bistro_gpu"]():
        tests.run(
            "test/rt/gpu/test_bistro_gpu.mojo",
            gpu_test_bistro_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_bmw_materials_gpu"]():
        tests.run(
            "test/rt/gpu/test_bmw_materials_gpu.mojo",
            gpu_test_bmw_materials_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_coated_shading_gpu"]():
        tests.run(
            "test/rt/gpu/test_coated_shading_gpu.mojo",
            gpu_test_coated_shading_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_crown_gpu"]():
        tests.run(
            "test/rt/gpu/test_crown_gpu.mojo",
            gpu_test_crown_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_cwbvh_call_gpu"]():
        tests.run(
            "test/rt/gpu/test_cwbvh_call_gpu.mojo",
            gpu_test_cwbvh_call_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_delta_metal_gpu"]():
        tests.run(
            "test/rt/gpu/test_delta_metal_gpu.mojo",
            gpu_test_delta_metal_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_gpu"]():
        tests.run("test/rt/gpu/test_gpu.mojo", gpu_test_gpu.run_tests[True])
    comptime if module_selected["gpu_test_primary_launch_gpu"]():
        tests.run(
            "test/rt/gpu/test_primary_launch_gpu.mojo",
            gpu_test_primary_launch_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_ray_order_gpu"]():
        tests.run(
            "test/rt/gpu/test_ray_order_gpu.mojo",
            gpu_test_ray_order_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_target_capacity_gpu"]():
        tests.run(
            "test/rt/gpu/test_target_capacity_gpu.mojo",
            gpu_test_target_capacity_gpu.run_tests[True],
        )
    comptime if module_selected["gpu_test_wavefront_contract_gpu"]():
        tests.run(
            "test/rt/gpu/test_wavefront_contract_gpu.mojo",
            gpu_test_wavefront_contract_gpu.run_tests[True],
        )
    tests^.finish()
