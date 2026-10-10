"""The performance gate must reject misleading partial or invalid evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench/warp'))

spec = importlib.util.spec_from_file_location('gpu_threshold', Path(__file__).resolve().parents[2] / 'bench/warp/check_gpu_threshold.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GpuThresholdTests(unittest.TestCase):
    def rebuild_record(self):
        faces = 256
        scope = dict(resident_inputs=True, geometry_publication=True, face_bounds=True,
                     tree_rebuild=True, synchronization=True, owner_allocation=False,
                     graph_capture=False, input_changes=False, validation_download=False)
        record = dict(scope='resident indexed mesh rebuild', faces=faces, vertices=faces,
                      groups=4, winding=False, devices=['cuda'], repeats=101,
                      threshold_evidence=True, timing_scope=scope, validation=[], samples=[])
        for implementation in ('bajo', 'warp'):
            for mode in ('direct', 'graph'):
                record['samples'].append(dict(implementation=implementation, device='cuda',
                    phase='rebuild' if mode == 'direct' else 'rebuild-graph', ns=[100]*101))
                for phase in (0,1,2):
                    record['validation'].append(dict(implementation=implementation, mode=mode,
                        phase=phase, matched=True, independent=True, changed_input=phase != 0,
                        queries=8, candidate_entries=8*faces, complete_forest=True, nodes=4*faces-5))
        record['changed_order_timing'] = dict(target_phases=[0,1], reset_to_opposite_before_each=True,
            preparation_timed=False, input_address_retained=True, target_centroid_inversions={'0':1,'1':1})
        record['order_validation'] = [copy.deepcopy(r) for r in record['validation'] if r['phase'] in (0,1)]
        for implementation in ('bajo','warp'):
            for phase in sorted(gate.REBUILD_PHASES - {'rebuild','rebuild-graph'}):
                record['samples'].append(dict(implementation=implementation, device='cuda', phase=phase, ns=[100]*101))
        return record

    def test_rebuild_requires_timed_changed_order_evidence(self):
        for index in range(8):
            record = self.rebuild_record()
            del record['order_validation'][index]
            self.assertTrue(self.check_rebuild(record)[1],index)
        for phase in gate.REBUILD_PHASES - {'rebuild','rebuild-graph'}:
            record = self.rebuild_record()
            record['samples'] = [r for r in record['samples'] if r['phase'] != phase]
            self.assertTrue(self.check_rebuild(record)[1],phase)
        for field,value in [('target_phases',[0]),('reset_to_opposite_before_each',False),
                            ('preparation_timed',True),('input_address_retained',False),
                            ('target_centroid_inversions',{'0':0,'1':1})]:
            record = self.rebuild_record()
            record['changed_order_timing'][field] = value
            self.assertTrue(self.check_rebuild(record)[1],field)

    def check_rebuild(self, record):
        return gate.check_record(record, .9, check_sources=False,
            expected_phases={'rebuild','rebuild-graph'}, expected_rebuild_faces=256)

    def test_complete_indexed_rebuild_passes(self):
        self.assertFalse(self.check_rebuild(self.rebuild_record())[1])
        self.assertEqual(len(gate.REBUILD_REPORTS),5)

    def test_rebuild_requires_both_modes_and_all_changed_cases(self):
        for index in range(12):
            record = self.rebuild_record()
            del record['validation'][index]
            self.assertTrue(self.check_rebuild(record)[1],index)
        record = self.rebuild_record()
        record['validation'][0]['complete_forest'] = False
        self.assertTrue(self.check_rebuild(record)[1])
        record = self.rebuild_record()
        record['validation'][0]['candidate_entries'] -= 1
        self.assertTrue(self.check_rebuild(record)[1])

    def test_rebuild_rejects_refit_wrong_workload_and_scope(self):
        for field,value in [('scope','refit'),('faces',1024),('vertices',128),
                            ('groups',0),('winding',True),('threshold_evidence',False)]:
            record = self.rebuild_record()
            record[field] = value
            self.assertTrue(self.check_rebuild(record)[1],field)
        for field in self.rebuild_record()['timing_scope']:
            record = self.rebuild_record()
            record['timing_scope'][field] = not record['timing_scope'][field]
            self.assertTrue(self.check_rebuild(record)[1],field)

    def winding_rebuild_record(self):
        r = self.rebuild_record()
        r.update(scope='resident indexed mesh rebuild with winding metadata', winding=True,
                 repeats=501, warmup_ns=50_000_000, warmup_min_calls=2,
                 pending_registration=False)
        r['timing_scope']['winding_metadata_update'] = True
        r['reference_pipeline'] = dict(native_mesh_rebuild=False, shared_points=True,
            rebuilt_grouped_bvh=True, refitted_ungrouped_winding_mesh=True,
            extra_mesh_tree_refit_timed=True, support_winding_number=True,
            grouped_bvh_leaf_size=1, winding_mesh_leaf_size=4)
        for sample in r['samples']:
            sample['ns'] = [100]*501
        for field in ('validation','order_validation'):
            for row in r[field]:
                row.update(winding_queries=8, winding_matched=True,
                    analytic_positions_independent=True, retained_reference_addresses=True,
                    bajo_cache=dict(components=64, nodes=127, original_faces=256,
                        analytic_queries=8, retained_addresses=21, complete_cache=True, matched=True))
        return r

    def check_winding_rebuild(self, record, leaf=4):
        return gate.check_record(record, .9, check_sources=False,
            expected_phases=gate.REBUILD_PHASES, expected_winding_rebuild=(256,leaf))

    def test_winding_rebuild_requires_both_reference_configurations(self):
        self.assertEqual(len(gate.WINDING_REBUILD_REPORTS),10)
        r = self.winding_rebuild_record()
        self.assertFalse(self.check_winding_rebuild(r)[1])
        self.assertTrue(self.check_winding_rebuild(r,8)[1])
        r['reference_pipeline']['winding_mesh_leaf_size'] = 8
        self.assertFalse(self.check_winding_rebuild(r,8)[1])

    def test_winding_rebuild_rejects_partial_or_fabricated_reference(self):
        for field in self.winding_rebuild_record()['reference_pipeline']:
            r = self.winding_rebuild_record()
            del r['reference_pipeline'][field]
            self.assertTrue(self.check_winding_rebuild(r)[1],field)
        for field,value in [('winding',False),('pending_registration',True),('repeats',101),
                            ('warmup_ns',0),('threshold_evidence',False)]:
            r = self.winding_rebuild_record()
            r[field] = value
            self.assertTrue(self.check_winding_rebuild(r)[1],field)
        r = self.winding_rebuild_record()
        del r['timing_scope']['winding_metadata_update']
        self.assertTrue(self.check_winding_rebuild(r)[1])

    def test_winding_rebuild_requires_all_cache_and_analytic_checks(self):
        for field in ('validation','order_validation'):
            for index in range(len(self.winding_rebuild_record()[field])):
                for mutation in ('delete','duplicate','winding_queries','winding_matched',
                                 'analytic_positions_independent','retained_reference_addresses','bajo_cache'):
                    r = self.winding_rebuild_record()
                    row = r[field][index]
                    if mutation == 'delete':
                        del r[field][index]
                    elif mutation == 'duplicate':
                        r[field].append(copy.deepcopy(row))
                    elif mutation == 'bajo_cache':
                        if row['implementation'] != 'bajo':
                            continue
                        row['bajo_cache']['retained_addresses'] = 3
                    elif mutation == 'retained_reference_addresses':
                        if row['implementation'] != 'warp':
                            continue
                        row[mutation] = False
                    else:
                        row[mutation] = False
                    self.assertTrue(self.check_winding_rebuild(r)[1],(field,index,mutation))

    def construction_record(self):
        faces = 256
        record = dict(repeats=101, faces=faces, vertices=faces, queries=8, devices=['cuda'],
                      construction='complete host-input pipeline',
                      construction_modes={'build-grouped': {'groups': 4, 'winding': False},
                                          'build-winding': {'groups': 0, 'winding': True}},
                      timing_scope={'fixture_generation': False, 'geometry_upload': True,
                                    'host_tree_construction': True, 'scratch_allocation': True,
                                    'synchronization': True, 'owner_destruction': False,
                                    'validation_download': False}, samples=[], validation=[])
        for phase, winding in [('build-grouped', False), ('build-winding', True)]:
            for implementation, ns in [('bajo', 100), ('warp', 90)]:
                record['samples'].append(dict(device='cuda', phase=phase, implementation=implementation, ns=[ns]*101))
                record['validation'].append(dict(device='cuda', phase=phase, implementation=implementation,
                                                 matched=True, independent=True, components=faces*3,
                                                 original_faces=faces, independent_queries=8,
                                                 groups=0 if winding else 4, winding_queries=8 if winding else 0,
                                                 nodes=(2*faces-1 if winding else 4*faces-5) if implementation == 'bajo' else None))
        return record

    def check_construction(self, record):
        return gate.check_record(record, .9, check_sources=False,
                                 expected_phases={'build-grouped', 'build-winding'},
                                 expected_construction_faces=256)

    def test_complete_construction_evidence_passes(self):
        self.assertFalse(self.check_construction(self.construction_record())[1])

    def test_construction_rejects_upload_only_or_wrong_workload(self):
        for field, value in [('faces', 1024), ('vertices', 64), ('queries', 4),
                             ('construction', 'upload prebuilt tree'), ('devices', ['cpu'])]:
            record = self.construction_record()
            record[field] = value
            self.assertTrue(self.check_construction(record)[1], field)
        for field in ('geometry_upload', 'host_tree_construction', 'scratch_allocation', 'synchronization'):
            record = self.construction_record()
            record['timing_scope'][field] = False
            self.assertTrue(self.check_construction(record)[1], field)

    def test_construction_requires_full_geometry_forest_and_winding_checks(self):
        for row in range(4):
            for field, value in [('original_faces', 128), ('components', 128), ('independent_queries', 4),
                                 ('groups', 1), ('matched', False), ('independent', False)]:
                record = self.construction_record()
                record['validation'][row][field] = value
                self.assertTrue(self.check_construction(record)[1], (row, field))
        for row in (0, 2):
            record = self.construction_record()
            record['validation'][row]['nodes'] -= 1
            self.assertTrue(self.check_construction(record)[1])
        record = self.construction_record()
        record['validation'][3]['winding_queries'] = 0
        self.assertTrue(self.check_construction(record)[1])

    def test_construction_requires_both_modes_and_all_five_workloads(self):
        record = self.construction_record()
        record['samples'] = record['samples'][:2]
        self.assertTrue(self.check_construction(record)[1])
        for faces in (256, 1024, 4096, 16384, 65536):
            self.assertEqual(gate.REPORTS[f'indexed-construction-{faces}-cuda'], {'build-grouped', 'build-winding'})

    def setUp(self):
        self.record = dict(repeats=101, validation=[dict(device='cuda', phase='query-0-0', matched=True, independent=True)],
                           samples=[dict(device='cuda', phase='query-0-0', implementation='bajo', ns=[100] * 101),
                                    dict(device='cuda', phase='query-0-0', implementation='warp', ns=[90] * 101)])

    def check(self, record=None, expected=None):
        return gate.check_record(self.record if record is None else record, .9, check_sources=False, expected_phases=expected)

    def test_tuned_launch_workloads_are_required(self):
        self.assertEqual(gate.QUERY_LAUNCHES['mesh-1024-q256-mask199-cuda'], (8, 8))
        self.assertEqual(gate.QUERY_LAUNCHES['mesh-1024-q256-mask32-cuda'], (64, 8))
        self.assertTrue({'query-0-0', 'query-2-0'} <= gate.REPORTS['mesh-1024-q256-mask199-cuda'])
        self.assertTrue({'query-5-0', 'query-5-1'} <= gate.REPORTS['mesh-1024-q256-mask32-cuda'])

    def test_tuned_launch_evidence_rejects_missing_or_slower_launches(self):
        self.record.update(bajo_block_size=8, warp_block_size=8)
        def check():
            return gate.check_record(self.record, .9, check_sources=False, expected_query_blocks=(8, 8))[1]
        self.assertFalse(check())
        for field in ('bajo_block_size', 'warp_block_size'):
            self.record[field] = 256
            self.assertTrue(check())
            self.record.pop(field)
            self.assertTrue(check())
            self.record[field] = 8

    def test_boundary_passes_and_lower_ratio_fails(self):
        self.assertFalse(self.check()[1])
        self.record['samples'][1]['ns'] = [89] * 101
        self.assertTrue(self.check()[1])

    def test_invalid_samples_never_pass(self):
        for value in (float('nan'), float('inf'), 0, -1):
            with self.subTest(value=value):
                record = copy.deepcopy(self.record)
                record['samples'][0]['ns'][0] = value
                self.assertTrue(self.check(record)[1])
        self.record['samples'][0]['ns'] = [100]
        self.assertTrue(self.check()[1])

    def test_missing_phase_or_implementation_fails(self):
        self.assertTrue(self.check(expected={'query-0-0', 'query-1-0'})[1])
        self.record['samples'].pop()
        self.assertTrue(self.check()[1])

    def test_unstable_or_duplicate_samples_fail(self):
        record = copy.deepcopy(self.record)
        record['samples'][0]['ns'][:20] = [400] * 20
        self.assertTrue(self.check(record)[1])
        self.record['samples'].append(copy.deepcopy(self.record['samples'][0]))
        self.assertTrue(self.check()[1])

    def test_isolated_stall_does_not_invalidate_large_sample_set(self):
        self.record['samples'][0]['ns'][0] = 400
        rows, failures = self.check()
        self.assertFalse(failures)
        self.assertEqual(rows[0][1], .9)

    def test_threshold_requires_confident_median_ratio(self):
        self.record['samples'][1]['ns'] = [89] * 50 + [91] + [93] * 50
        self.assertGreater(self.check()[0][0][1], .9)
        self.assertTrue(self.check()[1])

    def test_failed_or_cpu_only_correctness_fails(self):
        self.record['validation'][0]['matched'] = False
        self.assertTrue(self.check()[1])
        self.record['validation'][0]['matched'] = True
        self.record['validation'][0]['device'] = 'cpu'
        self.assertTrue(self.check()[1])

    def test_mixed_or_unidentified_warp_baseline_fails(self):
        _, failures = gate.check_record(self.record, .9, check_sources=False, expected_warp_commit='new-warp')
        self.assertTrue(failures)
        self.record['warp_reference'] = dict(commit='old-warp', native_hashes={'bin/warp.so': 'test'}, source_hashes={'native/bvh.h': 'test'})
        self.assertTrue(gate.check_reference(self.record, 'new-warp', check_files=False))
        self.assertFalse(gate.check_reference(self.record, 'old-warp', check_files=False))

    def test_graph_requires_both_changed_input_checks_and_exact_configuration(self):
        self.record.update(refit_graph=True, refit_winding=False)
        self.record['validation'].extend(
            dict(device='cuda', phase='refit', implementation=implementation,
                 changed_inputs=True, matched=True, independent=True)
            for implementation in ('bajo', 'warp')
        )
        def check():
            return gate.check_record(self.record, .9, check_sources=False, expected_graph_winding=False)[1]
        self.assertFalse(check())
        self.record['validation'][-1]['changed_inputs'] = False
        self.assertTrue(check())
        self.record['validation'][-1]['changed_inputs'] = True
        self.record['refit_winding'] = True
        self.assertTrue(check())
        self.record['refit_winding'] = False
        self.record['refit_graph'] = False
        self.assertTrue(check())

    def test_bounds_graph_requires_refit_rebuild_and_full_changed_input_checks(self):
        self.record.update(update_graph=True, devices=['cuda'], items=256, queries=512, groups=4)
        for phase in ('update-0', 'update-1'):
            for implementation in ('bajo', 'warp'):
                self.record['validation'].append(dict(device='cuda', phase=phase, implementation=implementation,
                    changed_inputs=True, matched=True, independent=True, items=256,
                    independent_queries=4096, nodes=1019 if implementation == 'bajo' else None))
        def check():
            return gate.check_record(self.record, .9, check_sources=False, expected_bounds_graph=True)[1]
        self.assertFalse(check())
        for key, value in [('changed_inputs', False), ('independent_queries', 512), ('items', 255)]:
            original = self.record['validation'][-1][key]
            self.record['validation'][-1][key] = value
            self.assertTrue(check())
            self.record['validation'][-1][key] = original
        self.record['validation'][-2]['nodes'] = 511
        self.assertTrue(check())
        self.record['validation'][-2]['nodes'] = 1019
        self.record['update_graph'] = False
        self.assertTrue(check())
        self.record['update_graph'] = True
        self.record['devices'] = ['cpu', 'cuda']
        self.assertTrue(check())

    def test_velocity_updates_require_full_changed_components_and_interpolation(self):
        self.record.update(velocity_update=True, vertices=3, queries=1)
        for phase in ('velocity-update', 'velocity-clear'):
            for implementation in ('bajo', 'warp'):
                self.record['validation'].append(dict(device='cuda', phase=phase, implementation=implementation,
                    changed_inputs=True, matched=True, independent=True, components=9, independent_queries=1))
        def check():
            return gate.check_record(self.record, .9, check_sources=False, expected_velocity_update=True)[1]
        self.assertFalse(check())
        self.record['validation'][-1]['components'] = 6
        self.assertTrue(check())
        self.record['validation'][-1]['components'] = 9
        self.record['validation'][-1]['independent_queries'] = 0
        self.assertTrue(check())
        self.record['validation'][-1]['independent_queries'] = 1
        self.record['velocity_update'] = False
        self.assertTrue(check())


class RenderingThresholdTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(repeats=101, warmup_ns=50_000_000, warmup_min_calls=2,
            warp_reference=dict(commit='test-warp', native_hashes={'bin/warp.so': 'test'},
                                source_hashes={'native/bvh.h': 'test'}), meshes=[], validation=[])
        for scene in ('bunny', 'dragon'):
            configurations = [('bajo', f'cuda-hploc-cwbvh8-m{leaf}') for leaf in (1, 3)]
            configurations += [('warp', f'cuda:0-{method}-l{leaf}')
                               for method in ('lbvh', 'sah', 'cubql') for leaf in (1, 4)]
            for implementation, label in configurations:
                ns = [100 if implementation == 'bajo' else 90] * 101
                self.record['meshes'].append(dict(scene=scene, implementation=implementation,
                    label=label, triangles=100, rays=512 * 288, build_ns=ns.copy(),
                    trace_ns=ns.copy(), any_ns=ns.copy()))
                self.record['validation'].append(dict(scene=scene, implementation=implementation,
                    label='warp-' + label if implementation == 'warp' else label,
                    rays=512 * 288, reference='warp-cpu-sah-l4', validated=True,
                    hit_disagreements=0, any_hit_disagreements=0, distance_disagreements=0))

    def check(self):
        return gate.check_rendering(self.record, .9, 'test-warp', check_sources=False)

    def test_all_configurations_and_boundary_pass(self):
        rows, failures = self.check()
        self.assertEqual(len(rows), 6)
        self.assertFalse(failures)
        for row in self.record['meshes']:
            if row['implementation'] == 'warp':
                row['build_ns'] = [89] * 101
        self.assertTrue(self.check()[1])

    def test_missing_scene_configuration_or_warp_validation_fails(self):
        pristine = copy.deepcopy(self.record)
        self.record['meshes'].pop()
        self.assertTrue(self.check()[1])
        self.record = copy.deepcopy(pristine)
        self.record['validation'].pop()
        self.assertTrue(self.check()[1])
        self.record = copy.deepcopy(pristine)
        self.record['meshes'] = [r for r in self.record['meshes'] if r['scene'] != 'dragon']
        self.assertTrue(self.check()[1])

    def test_bad_samples_warmup_counts_and_geometry_fail(self):
        pristine = copy.deepcopy(self.record)
        for key, value in (('repeats', 11), ('warmup_ns', 0), ('warmup_min_calls', 0)):
            self.record = copy.deepcopy(pristine)
            self.record[key] = value
            self.assertTrue(self.check()[1])
        self.record = copy.deepcopy(pristine)
        self.record['meshes'][0]['rays'] -= 1
        self.assertTrue(self.check()[1])
        self.record = copy.deepcopy(pristine)
        self.record['meshes'][0]['build_ns'][:20] = [400] * 20
        self.assertTrue(self.check()[1])

    def test_mask_and_distance_disagreements_fail(self):
        for key in ('hit_disagreements', 'any_hit_disagreements', 'distance_disagreements'):
            self.record['validation'][-1][key] = 1
            self.assertTrue(self.check()[1])
            self.record['validation'][-1][key] = 0

    def test_missing_disagreement_counts_fail(self):
        self.record['validation'][-1].pop('distance_disagreements')
        self.assertTrue(self.check()[1])

    def test_duplicate_gpu_configuration_fails(self):
        self.record['meshes'].append(copy.deepcopy(self.record['meshes'][0]))
        self.assertTrue(self.check()[1])


class CooperativeMeshEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(repeats=101, devices=['cuda'], faces=256, queries=3,
                           groups=4, bajo_cooperative=True, bajo_cooperative_block_size=32,
                           warp_cooperative_aabb="tile_mesh_query_aabb", warp_cooperative_block_size=32,
                           warp_cooperative_group_strategy="separate owners with original face mapping",
                           query_kinds=[9], independent_query_indices=[0, 1, 2],
                           validation=[dict(device='cuda', phase=phase, matched=True, independent=True,
                                            matched_queries=3, independent_queries=3)
                                       for phase in ('query-9-0', 'query-9-1')],
                           samples=[dict(device='cuda', phase=phase, implementation=implementation,
                                         ns=[100 if implementation == 'bajo' else 120] * 101)
                                    for phase in ('refit', 'query-9-0', 'query-9-1')
                                    for implementation in ('bajo', 'warp')])

    def check(self):
        return gate.check_record(self.record, .9, check_sources=False,
                                 expected_phases={'refit', 'query-9-0', 'query-9-1'},
                                 expected_cooperative_aabb=True, expected_mesh_workload=(256, 3))[1]

    def test_complete_collective_candidates_pass(self):
        self.assertFalse(self.check())

    def test_scalar_wrong_launch_or_workload_cannot_substitute(self):
        for field, value in [('bajo_cooperative', False), ('bajo_cooperative_block_size', 64),
                             ('devices', ['cpu']), ('query_kinds', [8]), ('groups', 2),
                             ('faces', 1024), ('queries', 2), ('warp_cooperative_aabb', 'scalar'),
                             ('warp_cooperative_block_size', 64), ('warp_cooperative_group_strategy', 'postfilter')]:
            with self.subTest(field=field):
                original = self.record[field]
                self.record[field] = value
                self.assertTrue(self.check())
                self.record[field] = original

    def test_partial_or_missing_independent_candidates_fail(self):
        self.record['independent_query_indices'] = [0, 2]
        self.assertTrue(self.check())
        self.record['independent_query_indices'] = [0, 1, 2]
        for evidence in self.record['validation']:
            for field in ('matched_queries', 'independent_queries'):
                with self.subTest(phase=evidence['phase'], field=field):
                    evidence[field] = 2
                    self.assertTrue(self.check())
                    evidence[field] = 3
        self.record['validation'].pop()
        self.assertTrue(self.check())


if __name__ == '__main__':
    unittest.main()
