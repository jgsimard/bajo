from std.testing import TestSuite, assert_equal, assert_true

from examples.stress_scenes import (
    make_indirect_hall_world,
    make_many_lights_world,
    make_specular_transport_world,
)


def test_many_lights_scene_has_dense_geometry_and_light_population() raises:
    var world = make_many_lights_world()
    assert_equal(len(world.scene_data().spheres()), 528)
    assert_equal(len(world.scene_data().triangle_vertices()) / 3, 10)
    assert_equal(len(world.scene_data().lights().records), 96)
    assert_true(len(world.scene_data().surfaces().dielectrics) > 0)
    assert_true(len(world.scene_data().surfaces().metals) > 0)


def test_indirect_hall_hides_multiple_triangle_emitters_behind_baffles() raises:
    var world = make_indirect_hall_world()
    assert_equal(len(world.scene_data().spheres()), 8)
    assert_equal(len(world.scene_data().triangle_vertices()) / 3, 62)
    assert_equal(len(world.scene_data().lights().records), 4)


def test_specular_transport_scene_mixes_delta_paths_and_small_lights() raises:
    var world = make_specular_transport_world()
    assert_equal(len(world.scene_data().spheres()), 43)
    assert_equal(len(world.scene_data().triangle_vertices()) / 3, 36)
    assert_equal(len(world.scene_data().lights().records), 5)
    assert_equal(len(world.scene_data().surfaces().dielectrics), 2)
    assert_equal(len(world.scene_data().surfaces().metals), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
