import unittest
import ast
import json
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace
from typing import List, Union
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from research.branching_validity import TOLERANCES, capture, capture_rngs, clone_tree, compare, quaternion_error, restore, restore_rngs


def trace():
    result = {key: np.zeros((3, 2, 3)) for key in TOLERANCES if key != 'rotation'}
    result['quaternion'] = np.tile([1., 0, 0, 0], (3, 2, 3, 1))
    result['flags'] = np.zeros((2, 2, 7), dtype=np.int8)
    return result


class BranchValidityTests(unittest.TestCase):
    def test_archive_analysis_identifies_delayed_drift(self):
        from scripts.analyze_branching_validity import analyze
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = trace()
            np.savez_compressed(root/'parent-2-reference.npz', **reference)
            rows = []
            for mode in ('physical', 'augmented'):
                replay = trace()
                if mode == 'physical':
                    replay['qvel'][1, 0, 0] = .02
                np.savez_compressed(root/f'parent-2-{mode}-a0-r0.npz', **replay)
                rows.append(dict(**compare(reference, replay), parent=2, mode=mode, anchor=0, repeat=0,
                                 initial_errors={key: 0. for key in TOLERANCES if key != 'rotation'}))
            (root/'report.json').write_text(json.dumps(dict(state='complete', comparisons=rows,
                gate_passed=dict(physical=False, augmented=True), transitions_completed=123,
                wall_seconds=2, decision='fixture')))
            result = analyze(root)
            self.assertEqual(result['comparisons'][0]['first_physical_failure_per_lane'], [1, -1])
            self.assertEqual(result['summaries']['physical']['initial_failed_lanes'], 0)
            self.assertEqual(result['summaries']['physical']['physical_failed_lanes'], 1)
            self.assertEqual(result['summaries']['augmented']['physical_failed_lanes'], 0)

    def test_actual_pinned_batched_rng_restoration_without_simulator_import(self):
        # Execute only the installed class definition; importing mani_skill on
        # macOS initializes graphics. This is the real wrapper, not a mock.
        path = Path(distribution('mani_skill').locate_file('mani_skill/envs/utils/randomization/batched_rng.py'))
        source = ast.parse(path.read_text())
        definition = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'BatchedRNG')
        scope = dict(np=np, List=List, Union=Union, common=SimpleNamespace(to_numpy=np.asarray))
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), scope)
        batch = scope['BatchedRNG'].from_seeds([1, 2, 3])
        env = SimpleNamespace(_batched_episode_rng=batch, _episode_rng=batch.rngs[0])
        saved = capture_rngs(env)
        expected = batch.rand(3)
        restore_rngs(env, saved)
        np.testing.assert_array_equal(batch.rand(3), expected)
        self.assertIs(env._episode_rng, batch.rngs[0])
        restore_rngs(env, saved)
        np.testing.assert_array_equal(batch.rand(3), expected)

    def test_augmented_restore_recovers_buffers_drives_clock_and_rng(self):
        class Buffer:
            def __init__(self, value): self.value = value
            def torch(self): return self.value
        class Robot:
            active_joints = ['a', 'b']
            _data_index = torch.arange(2, dtype=torch.int32)
            def __init__(self, px): self.px, self.qf = px, torch.ones(2, 2)
            def get_joint_target_indices(self, indices):
                return torch.meshgrid(self._data_index, indices, indexing='ij')
            def get_qf(self): return self.qf
            def set_qf(self, value): self.qf = value
            def set_joint_drive_targets(self, value, joints, indices):
                self.px.cuda_articulation_target_qpos.value[self.get_joint_target_indices(indices)] = value
            def set_joint_drive_velocity_targets(self, value, joints, indices):
                self.px.cuda_articulation_target_qvel.value[self.get_joint_target_indices(indices)] = value
        class Env:
            device = torch.device('cpu')
            def __init__(self):
                self.physical = {'actors': {'cube': torch.ones(2, 13)}}
                self._elapsed_steps = torch.tensor([15, 15])
                self._episode_rng = np.random.RandomState(4)
                self._batched_episode_rng = self._episode_rng
                px = SimpleNamespace(cuda_articulation_target_qpos=Buffer(torch.full((2, 2), 3.)),
                                     cuda_articulation_target_qvel=Buffer(torch.full((2, 2), 4.)),
                                     gpu_update_articulation_kinematics=lambda: None)
                self.scene = SimpleNamespace(px=px, _gpu_apply_all=lambda: None, _gpu_fetch_all=lambda: None)
                c = SimpleNamespace(_step=5, _target_qpos=torch.full((2, 2), 7.))
                self.agent = SimpleNamespace(robot=Robot(px), controller=SimpleNamespace(controllers={'arm': c}))
            def get_state_dict(self): return self.physical
            def set_state_dict(self, state): self.physical = state
        env = Env()
        with patch('torch.cuda.get_rng_state_all', return_value=[]), patch('torch.cuda.set_rng_state_all'):
            snapshot = capture(env)
            expected_random = env._episode_rng.rand()
            env.agent.controller.controllers['arm']._target_qpos.fill_(99)
            env.agent.controller.controllers['arm']._step = 90
            env._elapsed_steps.fill_(49)
            env.agent.robot.px.cuda_articulation_target_qpos.value.fill_(80)
            env.agent.robot.qf.fill_(5)
            env.physical['actors']['cube'].fill_(6)
            restore(env, snapshot, augmented=True)
            self.assertEqual(env.agent.controller.controllers['arm']._step, 5)
            self.assertTrue(torch.all(env.agent.controller.controllers['arm']._target_qpos == 7))
            self.assertTrue(torch.all(env.agent.robot.px.cuda_articulation_target_qpos.value == 3))
            self.assertTrue(torch.all(env.agent.robot.px.cuda_articulation_target_qvel.value == 4))
            self.assertTrue(torch.all(env.agent.robot.qf == 1))
            self.assertTrue(torch.all(env._elapsed_steps == 15))
            self.assertEqual(env._episode_rng.rand(), expected_random)
            self.assertIs(env._episode_rng, env._batched_episode_rng)
            self.assertTrue(torch.all(env.physical['actors']['cube'] == 1))
            env.physical['actors']['cube'].fill_(9)
            self.assertTrue(torch.all(snapshot['physical']['actors']['cube'] == 1))

    def test_copies_tensor_views_without_aliasing(self):
        original = {'a': torch.ones(3), 'nested': {'b': np.ones(2)}}
        snapshot = clone_tree(original)
        original['a'][0] = 9
        original['nested']['b'][0] = 8
        self.assertEqual(float(snapshot['a'][0]), 1.)
        self.assertEqual(snapshot['nested']['b'][0], 1.)

    def test_quaternion_sign_and_small_rotation(self):
        self.assertEqual(float(quaternion_error([1, 0, 0, 0], [-1, 0, 0, 0])), 0.)
        theta = 1e-5
        self.assertAlmostEqual(float(quaternion_error([1, 0, 0, 0],
                               [np.cos(theta/2), 0, 0, np.sin(theta/2)])), theta)

    def test_gate_detects_physical_and_success_changes(self):
        a, b = trace(), trace()
        self.assertTrue(compare(a, b)['passed'])
        b['position'][1, 0, 0] = 2*TOLERANCES['position']
        self.assertFalse(compare(a, b)['passed'])
        b = trace()
        b['flags'][-1, 0, 6] = 1
        result = compare(a, b)
        self.assertFalse(result['passed'])
        self.assertEqual(result['final_success_changed_lanes'], 1)

    def test_invalid_traces_fail_closed(self):
        a, b = trace(), trace()
        b['qvel'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare(a, b)


if __name__ == '__main__':
    unittest.main()
