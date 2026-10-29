import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from automanip.task_spec import TASK
from automanip.policy_io import load_policy, PolicySchema
from automanip.metrics import pickcube_success, PlacementSample, PlacementTracker, success_interval


class TaskTests(unittest.TestCase):
    def test_frozen(self):
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            TASK.horizon = 100
        self.assertEqual(len(TASK.fingerprint), 64)
        self.assertEqual(TASK.env_kwargs()["control_mode"], "pd_ee_delta_pose")

    def test_backend_constraints(self):
        for kwargs in ({"num_envs": 2}, {"backend": "auto"}, {"num_envs": True}):
            with self.assertRaises(ValueError):
                TASK.env_kwargs(**kwargs)
        self.assertEqual(TASK.env_kwargs(num_envs=4, backend="physx_cuda")["num_envs"], 4)


class MetricTests(unittest.TestCase):
    def test_stock_boundary_and_static_rule(self):
        self.assertTrue(pickcube_success(0.025, (0.2,) * 7))
        self.assertFalse(pickcube_success(0.025001, (0.0,) * 7))
        self.assertFalse(pickcube_success(0.0, (-0.201,) * 7))
        with self.assertRaises(ValueError):
            pickcube_success(math.nan, (0.0,) * 7)

    def test_release_support_and_consecutive_settling(self):
        tracker = PlacementTracker(2)
        good = PlacementSample(True, True, False, 0.0, 0.0)
        self.assertFalse(tracker.update(good))
        self.assertFalse(tracker.update(PlacementSample(True, True, True, 0.0, 0.0)))
        self.assertFalse(tracker.update(good))
        self.assertTrue(tracker.update(good))
        self.assertFalse(tracker.update(PlacementSample(True, False, False, 0.0, 0.0)))
        self.assertFalse(tracker.update(PlacementSample(True, True, False, 0.02, 0.0)))
        self.assertFalse(tracker.update(PlacementSample(False, True, False, 0.0, 0.0)))

    def test_invalid_placement_measurements(self):
        with self.assertRaises(ValueError):
            PlacementTracker().update(PlacementSample(True, True, False, math.inf, 0))

    def test_uncertainty(self):
        low, high = success_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(low + high, 1)
        self.assertGreater(success_interval(0, 10)[1], 0)
        with self.assertRaises(ValueError):
            success_interval(0, 0)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        # Deliberately synthetic dimensions; not a simulator observation claim.
        self.schema = PolicySchema(TASK.fingerprint, 3, 7)
        self.weights = {
            "layers.0.weight": {"shape": [2, 3], "values": [0.1]*6},
            "layers.0.bias": {"shape": [2], "values": [0.0]*2},
            "layers.1.weight": {"shape": [7, 2], "values": [0.1]*14},
            "layers.1.bias": {"shape": [7], "values": [0.0]*7},
        }
        self.manifest = dict(format="automanip-json-f32-v1", task_fingerprint=TASK.fingerprint,
                             observation_dim=3, action_dim=7, hidden_sizes=[2],
                             activation="tanh", weights_sha256="")

    def write(self):
        raw = json.dumps(self.weights).encode()
        self.manifest["weights_sha256"] = hashlib.sha256(raw).hexdigest()
        (self.directory / "weights.json").write_bytes(raw)
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def reject(self):
        self.write()
        with self.assertRaises(ValueError):
            load_policy(self.directory, self.schema)

    def test_valid_roundtrip_and_immutable_values(self):
        self.write()
        policy = load_policy(self.directory, self.schema)
        self.assertEqual(policy.hidden_sizes, (2,))
        self.assertAlmostEqual(policy.tensors["layers.0.weight"][0], 0.1)
        with self.assertRaises(TypeError):
            policy.tensors["injected"] = ()

    def test_interface_and_architecture_rejections(self):
        for field, value in (("observation_dim", 4), ("action_dim", True),
                             ("activation", "import me"), ("hidden_sizes", [257]),
                             ("hidden_sizes", [True]), ("task_fingerprint", "wrong")):
            with self.subTest(field=field, value=value):
                original = self.manifest[field]
                self.manifest[field] = value
                self.reject()
                self.manifest[field] = original

    def test_no_candidate_code_or_paths(self):
        self.manifest["module"] = "train.py"
        self.reject()

    def test_tensor_names_shapes_lengths_and_values(self):
        for replacement in ({"shape": [3, 2], "values": [0]*6},
                            {"shape": [2, 3], "values": [0]},
                            {"shape": [2, 3], "values": [True]*6},
                            {"shape": [2, 3], "values": [1e100]*6},
                            {"shape": [2, 3], "values": [math.nan]*6}):
            with self.subTest(replacement=replacement):
                self.weights["layers.0.weight"] = replacement
                self.reject()

    def test_digest_tampering(self):
        self.write()
        with (self.directory / "weights.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaises(ValueError):
            load_policy(self.directory, self.schema)

    def test_duplicate_keys_and_resource_limits(self):
        self.write()
        (self.directory / "manifest.json").write_text('{"format":1,"format":2}')
        with self.assertRaises(ValueError):
            load_policy(self.directory, self.schema)
        (self.directory / "manifest.json").write_bytes(b" " * 8193)
        with self.assertRaises(ValueError):
            load_policy(self.directory, self.schema)

    def test_parameter_budget(self):
        self.schema = PolicySchema(TASK.fingerprint, 4096, 7)
        self.manifest["observation_dim"] = 4096
        self.manifest["hidden_sizes"] = [256]
        self.reject()


class EvaluatorContractTests(unittest.TestCase):
    def test_actions(self):
        from automanip.evaluator import validate_actions
        validate_actions([[0.0]*7], batch_size=1)
        for actions in ([[math.nan]*7], [[1.01]*7], [[0.0]*6], []):
            with self.assertRaises(ValueError):
                validate_actions(actions, batch_size=1)

    def test_crashes_and_truncations(self):
        from automanip.evaluator import EpisodeOutcome
        self.assertFalse(EpisodeOutcome(True, True, False, False, True).success_at_end)
        self.assertFalse(EpisodeOutcome(False, True, False, True).success_at_end)
        self.assertTrue(EpisodeOutcome(True, True, False, True).success_at_end)


if __name__ == "__main__":
    unittest.main()
