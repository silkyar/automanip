"""Synthetic arrays only: no environments, deployment code or scored episodes."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from automanip.cumulative_episodes import (
    EpisodeRecorder, LEGACY_COLLECTOR_SHA256, import_legacy_demos, load_episodes,
)


TASK, INTERFACE = "stack-variation-A-v1", "panda-joint-state48-action8-v1"


def observations(*values):
    return np.asarray([np.full(48, value, dtype=np.float32) for value in values])


def recorder(path, count=1, auto=True, partition="training", behavior=None):
    return EpisodeRecorder(path, TASK, INTERFACE, partition,
                           {"actor_sha256": "synthetic-actor", "update": 0} if behavior is None else behavior,
                           count, auto)


def one_episode(path, partition="training"):
    record = recorder(path, auto=False, partition=partition)
    first, last = observations(0), observations(1)
    record.start(first, seed=11)
    record.step(first, np.zeros((1, 8)), last, [1.], [False], [True], {"success": [False]})
    record.finish("done")


def legacy_fixture(path, *, partition="public-training", successes=4, mixed_seed=False):
    """Synthetic arrays carrying the pinned reader's format, never real demos."""
    from research.discovery.amp_v1 import DATA_FORMAT, PRODUCTION_PINS
    path.mkdir()
    obs = np.arange(5*51*48, dtype=np.float32).reshape(5, 51, 48)/1000
    actions = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None]/10, (5, 50, 8)).copy()
    rewards = np.arange(250, dtype=np.float32).reshape(5, 50)/100
    final = np.arange(5) < successes
    seeds = np.full(5, 81250000, dtype=np.int64)
    if mixed_seed:
        seeds[-1] += 1
    np.savez_compressed(path/"trajectories.npz", observations=obs, actions=actions,
                        task_rewards=rewards, final_success=final, public_training_seed=seeds)
    metadata = dict(format=DATA_FORMAT, sha256=hashlib.sha256((path/"trajectories.npz").read_bytes()).hexdigest(),
                    env_id="StackCube-v1", observation_dim=48, action_dim=8, horizon=50,
                    control_mode="pd_joint_delta_pos", obs_mode="state", reward_mode="normalized_dense",
                    simulation_backend="physx_cuda", source_partition=partition, versions=PRODUCTION_PINS,
                    completed_episodes=5, successful_episodes=int(final.sum()), num_envs=5,
                    collector_source_sha256=LEGACY_COLLECTOR_SHA256,
                    teacher_actor_weights_sha256="synthetic-fixture-not-a-policy", incomplete_transition_rows=0,
                    fixture_only=True)
    (path/"dataset.json").write_text(json.dumps(metadata))
    return obs, actions, rewards, final


class EpisodePersistenceTests(unittest.TestCase):
    def test_asynchronous_autoreset_uses_true_successor_and_retains_every_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"episodes"
            record = recorder(path, count=2)
            first = observations(0, 1)
            record.start(first, [100, 101])
            following = observations(100, 2)  # lane0 has reset, lane1 continues
            record.step(first, np.full((2, 8), .25), following, [1., 2.], [1, 0], [0, 0],
                        {"success": [True, True], "final_observation": observations(10, 2),
                         "_final_observation": [True, False], "final_info": {"success": [False, True]}})
            record.step(following, np.full((2, 8), -.5), observations(101, 200), [3., 4.], [0, 0], [0, 1],
                        {"final_observation": observations(101, 20), "_final_observation": [False, True],
                         "final_info": {"success": [False, True]}}, behavior={"actor_sha256": "synthetic-next", "update": 1})
            record.finish("budget_exhausted")
            episodes = load_episodes(path, TASK, INTERFACE, for_training=True)
            self.assertEqual(len(episodes), 3)
            by_id = {(e["metadata"]["lane"], e["metadata"]["ordinal"]): e for e in episodes}
            failed, partial, successful = by_id[0, 0], by_id[0, 1], by_id[1, 0]
            self.assertEqual(failed["metadata"]["outcome"], "failure")
            np.testing.assert_array_equal(failed["next_observations"], observations(10))
            self.assertNotEqual(float(failed["next_observations"][0, 0]), 100.)
            self.assertEqual(successful["metadata"]["outcome"], "success")
            np.testing.assert_array_equal(successful["observations"], observations(1, 2))
            np.testing.assert_array_equal(successful["next_observations"], observations(2, 20))
            self.assertEqual(successful["metadata"]["steps"], 2)
            self.assertEqual([s["start"] for s in successful["metadata"]["behavior_segments"]], [0, 1])
            self.assertEqual(partial["metadata"]["outcome"], "incomplete")
            self.assertEqual(partial["metadata"]["reason"], "budget_exhausted")
            self.assertIsNone(partial["metadata"]["reset_seed_input"])
            self.assertEqual(partial["metadata"]["seed_kind"], "autoreset_unknown")
            self.assertEqual(failed["metadata"]["reset_seed_input"], 100)
            self.assertEqual(len({e["metadata"]["episode_id"] for e in episodes}), 3)
            self.assertEqual(sum(len(e["actions"]) for e in episodes), 4)

    def test_existing_successor_helper_reused_and_reset_success_is_not_terminal_success(self):
        from research.discovery.amp_v1 import actual_successor
        with tempfile.TemporaryDirectory() as temporary:
            record = recorder(temporary)
            record.start(observations(0), 1)
            with patch("research.discovery.amp_v1.actual_successor", wraps=actual_successor) as successor:
                record.step(observations(0), np.zeros((1, 8)), observations(99), [0.], [0], [1],
                            {"success": [True], "final_observation": observations(1), "_final_observation": [True]})
                successor.assert_called_once()
            record.finish("done")
            episode = load_episodes(temporary, TASK, INTERFACE)[0]
            self.assertEqual(episode["metadata"]["outcome"], "unknown")
            self.assertFalse(episode["success_known"][0])

    def test_non_autoreset_and_explicit_reset_preserve_unfinished_lanes(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = recorder(temporary, count=2, auto=False)
            first, next_obs = observations(0, 10), observations(1, 11)
            record.start(first, 12)
            record.step(first, np.zeros((2, 8)), next_obs, [0., 0.], [1, 0], [0, 0], {"success": [False, False]})
            with self.assertRaisesRegex(ValueError, "start"):
                record.step(next_obs, np.zeros((2, 8)), observations(2, 12), [0., 0.], [0, 0], [0, 0], {})
            reset = observations(100, 200)
            record.start(reset, [21, 22])
            record.step(reset, np.zeros((2, 8)), observations(101, 201), [1., 1.], [0, 0], [1, 1], {"success": [True, False]})
            self.assertEqual(record.finish("finished"), [])
            episodes = load_episodes(temporary, TASK, INTERFACE)
            self.assertEqual(len(episodes), 4)
            partial = next(e for e in episodes if not e["metadata"]["complete"])
            self.assertEqual(partial["metadata"]["lane"], 1)
            self.assertEqual(partial["metadata"]["reason"], "explicit_reset")
            np.testing.assert_array_equal(partial["next_observations"], observations(11))
            with self.assertRaises(ValueError):
                record.start(reset, 1)

    def test_invalid_steps_are_not_partially_appended(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = recorder(temporary)
            first = observations(0)
            record.start(first, 1)
            base = dict(obs=first, executed_actions=np.zeros((1, 8)), following=observations(1), rewards=[0.],
                        terminated=[0], truncated=[0], infos={})
            cases = ({"executed_actions": np.full((1, 8), 1.01)},
                     {"executed_actions": np.full((1, 8), np.nextafter(1., np.inf))}, {"terminated": [2]},
                     {"rewards": [np.nan]}, {"obs": observations(7)}, {"following": np.zeros((1, 47))},
                     {"truncated": [1], "infos": {}},
                     {"truncated": [1], "infos": {"final_observation": observations(1), "_final_observation": [False]}})
            for change in cases:
                with self.subTest(change=list(change)):
                    with self.assertRaises(ValueError):
                        record.step(**(base | change))
            record.step(**base)
            record.finish("interrupted")
            episode = load_episodes(temporary, TASK, INTERFACE)[0]
            self.assertEqual(episode["metadata"]["steps"], 1)
            self.assertFalse(episode["metadata"]["complete"])

    def test_partition_and_task_interface_rejection(self):
        for partition in ("training", "development", "final"):
            with self.subTest(partition=partition), tempfile.TemporaryDirectory() as temporary:
                one_episode(temporary, partition)
                self.assertEqual(len(load_episodes(temporary, TASK, INTERFACE)), 1)
                if partition == "training":
                    self.assertEqual(len(load_episodes(temporary, TASK, INTERFACE, for_training=True)), 1)
                else:
                    with self.assertRaisesRegex(ValueError, "training"):
                        load_episodes(temporary, TASK, INTERFACE, for_training=True)
                for task, interface in (("other-task", INTERFACE), (TASK, "other-controller")):
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        load_episodes(temporary, task, interface)

    def test_lane_swap_and_episode_count_budget_reject_without_losing_accepted_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = recorder(temporary, count=2, auto=False)
            record.start(observations(0, 10), 1)
            with self.assertRaisesRegex(ValueError, "continuity"):
                record.step(observations(10, 0), np.zeros((2, 8)), observations(1, 11), [0., 0.], [0, 0], [0, 0], {})
            record.step(observations(0, 10), np.zeros((2, 8)), observations(1, 11), [0., 0.], [0, 0], [0, 0], {})
            with patch("automanip.cumulative_episodes.MAX_EPISODES", 1):
                with self.assertRaisesRegex(ValueError, "count"):
                    record.step(observations(1, 11), np.zeros((2, 8)), observations(2, 12), [0., 0.], [0, 0], [0, 0], {})
            record.finish("stop")
            episodes = load_episodes(temporary, TASK, INTERFACE)
            self.assertEqual([e["metadata"]["steps"] for e in episodes], [1, 1])

    def test_source_inputs_and_completed_files_are_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            behavior = {"asset": {"version": 1}}
            record = recorder(temporary, auto=False, behavior=behavior)
            first, after, action = observations(0), observations(1), np.zeros((1, 8))
            record.start(first, 3)
            record.step(first, action, after, [1.], [0], [0], {})
            behavior["asset"]["version"] = 8
            first[:] = 44; after[:] = 45; action[:] = .8
            record.finish("cap")
            before = {p.name: p.read_bytes() for p in Path(temporary).iterdir()}
            with self.assertRaises(FileExistsError):
                recorder(temporary)
            episode = load_episodes(temporary, TASK, INTERFACE)[0]
            self.assertEqual(episode["metadata"]["behavior_segments"][0]["behavior"]["asset"]["version"], 1)
            np.testing.assert_array_equal(episode["observations"], observations(0))
            np.testing.assert_array_equal(episode["next_observations"], observations(1))
            np.testing.assert_array_equal(episode["actions"], np.zeros((1, 8)))
            episode["observations"][:] = 77
            self.assertEqual(float(load_episodes(temporary, TASK, INTERFACE)[0]["observations"][0, 0]), 0.)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(temporary).iterdir()})

    def test_digest_and_index_binding_tampering(self):
        for target in ("data", "index", "partition", "orphan"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)
                one_episode(path)
                entry = next(path.glob("episode-*.json"))
                if target == "data":
                    data = entry.with_suffix(".npz")
                    raw = bytearray(data.read_bytes()); raw[-1] ^= 1; data.write_bytes(raw)
                elif target == "index":
                    manifest = path/"index.json"
                    value = json.loads(manifest.read_text()); value["behavior"] = {"changed": True}; manifest.write_text(json.dumps(value))
                elif target == "partition":
                    value = json.loads(entry.read_text()); value["partition"] = "final"; entry.write_text(json.dumps(value))
                else:
                    entry.unlink()
                with self.assertRaises(ValueError):
                    load_episodes(path, TASK, INTERFACE)

    def test_npz_object_header_and_compressed_oversize_rejected_before_array_loading(self):
        for attack in ("object", "huge_shape", "expanded_size"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)
                one_episode(path)
                entry = next(path.glob("episode-*.json")); data = entry.with_suffix(".npz")
                buffer = io.BytesIO()
                with zipfile.ZipFile(io.BytesIO(data.read_bytes())) as old, zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as new:
                    for name in old.namelist():
                        raw = old.read(name)
                        if name == "observations.npy":
                            if attack == "object":
                                fake = io.BytesIO(); np.save(fake, np.full((1, 48), "object", dtype=object)); raw = fake.getvalue()
                            elif attack == "huge_shape":
                                fake = io.BytesIO(); np.lib.format.write_array_header_2_0(fake, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 48)})
                                raw = fake.getvalue()
                            else:
                                raw = bytes(5*1024*1024)
                        new.writestr(name, raw)
                raw = buffer.getvalue(); data.write_bytes(raw)
                value = json.loads(entry.read_text()); value["sha256"] = hashlib.sha256(raw).hexdigest(); entry.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_episodes(path, TASK, INTERFACE)

    def test_length_budget_rejects_before_accepting_and_partial_can_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = recorder(temporary, auto=False)
            record.start(observations(0), 1)
            with patch("automanip.cumulative_episodes.MAX_STEPS", 2):
                for index in range(2):
                    record.step(observations(index), np.zeros((1, 8)), observations(index+1), [0.], [0], [0], {})
                with self.assertRaisesRegex(ValueError, "length"):
                    record.step(observations(2), np.zeros((1, 8)), observations(3), [0.], [0], [0], {})
                record.finish("length_budget")
            episodes = load_episodes(temporary, TASK, INTERFACE)
            self.assertEqual(episodes[0]["metadata"]["steps"], 2)
            self.assertEqual(episodes[0]["metadata"]["reason"], "length_budget")

    def test_empty_store_and_unknown_outcome_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = recorder(temporary, auto=False)
            record.start(observations(0), None)
            record.finish("no_steps")
            self.assertEqual(load_episodes(temporary, TASK, INTERFACE), [])
        with tempfile.TemporaryDirectory() as temporary:
            record = recorder(temporary, auto=False)
            record.start(observations(0), 7)
            record.step(observations(0), np.zeros((1, 8)), observations(1), [0.], [1], [1], {})
            record.finish("end")
            episode = load_episodes(temporary, TASK, INTERFACE)[0]
            self.assertEqual(episode["metadata"]["outcome"], "unknown")
            self.assertTrue(episode["terminated"][-1] and episode["truncated"][-1])


class LegacyEpisodeImportTests(unittest.TestCase):
    def test_all_five_episodes_imported_including_failure_and_bc_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); source, destination = base/"legacy", base/"episodes"
            obs, actions, rewards, final = legacy_fixture(source)
            originals = {p.name: p.read_bytes() for p in source.iterdir()}
            receipt = import_legacy_demos(source, destination, TASK, INTERFACE)
            episodes = load_episodes(destination, TASK, INTERFACE, for_training=True)
            self.assertEqual(len(episodes), 5)
            self.assertEqual(receipt["total_imported_episodes"], 5)
            self.assertEqual(receipt["total_imported_transitions"], 250)
            self.assertEqual(receipt["successes"], 4)
            self.assertEqual(receipt["failures"], 1)
            self.assertEqual(receipt["new_environment_transitions"], 0)
            self.assertEqual(receipt["new_teacher_data"], 0)
            self.assertTrue(receipt["legacy_reader_requires_at_least_four_successes"])
            for episode in episodes:
                index = episode["metadata"]["lane"]
                np.testing.assert_array_equal(episode["observations"], obs[index, :-1])
                np.testing.assert_array_equal(episode["next_observations"], obs[index, 1:])
                np.testing.assert_array_equal(episode["actions"], actions[index])
                np.testing.assert_array_equal(episode["rewards"], rewards[index])
                self.assertFalse(episode["terminated"].any())
                np.testing.assert_array_equal(episode["truncated"], np.arange(50) == 49)
                np.testing.assert_array_equal(episode["success_known"], np.arange(50) == 49)
                self.assertEqual(bool(episode["success"][-1]), bool(final[index]))
                self.assertEqual(episode["metadata"]["reset_seed_input"], 81250000)
                self.assertEqual(episode["metadata"]["seed_kind"], "batched_reset_seed_argument")
            self.assertEqual(originals, {p.name: p.read_bytes() for p in source.iterdir()})
            receipt_path = Path(receipt["receipt_path"])
            self.assertEqual(receipt_path.parent, destination.parent.resolve())
            self.assertEqual(json.loads(receipt_path.read_text()), receipt)
            expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}
            self.assertEqual(expected, receipt["immutable_file_sha256"])
            digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(receipt["data_digest"], digest)
            self.assertEqual(receipt["old_dataset_sha256"], hashlib.sha256(originals["trajectories.npz"]).hexdigest())
            with self.assertRaises(FileExistsError):
                import_legacy_demos(source, destination, TASK, INTERFACE)

    def test_multiple_explicit_legacy_batches_preserve_per_lane_order_and_seed_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary)/"legacy", Path(temporary)/"episodes"
            obs, actions, rewards, final = legacy_fixture(source)
            np.savez_compressed(source/"trajectories.npz", observations=obs, actions=actions,
                                task_rewards=rewards, final_success=final,
                                public_training_seed=np.arange(5, dtype=np.int64)+900)
            metadata = json.loads((source/"dataset.json").read_text())
            metadata.update(num_envs=1, sha256=hashlib.sha256((source/"trajectories.npz").read_bytes()).hexdigest())
            (source/"dataset.json").write_text(json.dumps(metadata))
            import_legacy_demos(source, destination, TASK, INTERFACE)
            episodes = load_episodes(destination, TASK, INTERFACE)
            self.assertEqual([e["metadata"]["ordinal"] for e in episodes], list(range(5)))
            self.assertEqual([e["metadata"]["reset_seed_input"] for e in episodes], list(range(900, 905)))
            for index, episode in enumerate(episodes):
                np.testing.assert_array_equal(episode["observations"], obs[index, :-1])

    def test_nontraining_legacy_partitions_and_unreviewed_collectors_rejected(self):
        for partition in ("development", "final"):
            with self.subTest(partition=partition), tempfile.TemporaryDirectory() as temporary:
                source, destination = Path(temporary)/"legacy", Path(temporary)/"episodes"
                legacy_fixture(source, partition=partition)
                with self.assertRaisesRegex(ValueError, "partition"):
                    import_legacy_demos(source, destination, TASK, INTERFACE)
                self.assertFalse(destination.exists())
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary)/"legacy", Path(temporary)/"episodes"
            legacy_fixture(source)
            with self.assertRaisesRegex(ValueError, "Unsupported collector"):
                import_legacy_demos(source, destination, TASK, INTERFACE, expected_collector_sha256="0"*64)
            metadata = json.loads((source/"dataset.json").read_text()); metadata["collector_source_sha256"] = "0"*64
            (source/"dataset.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "collector digest"):
                import_legacy_demos(source, destination, TASK, INTERFACE)
            self.assertFalse(destination.exists())

    def test_mixed_batch_seeds_and_existing_legacy_minimum_are_not_silently_relaxed(self):
        for options, message in ((dict(mixed_seed=True), "scalar reset"), (dict(successes=0), "At least4")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                source, destination = Path(temporary)/"legacy", Path(temporary)/"episodes"
                legacy_fixture(source, **options)
                with self.assertRaisesRegex(ValueError, message):
                    import_legacy_demos(source, destination, TASK, INTERFACE)
                self.assertFalse(destination.exists())

    def test_legacy_archive_size_preflight_happens_before_existing_reader(self):
        for constant in ("MAX_LEGACY_COMPRESSED_BYTES", "MAX_LEGACY_EXPANDED_BYTES"):
            with self.subTest(constant=constant), tempfile.TemporaryDirectory() as temporary:
                source, destination = Path(temporary)/"legacy", Path(temporary)/"episodes"
                legacy_fixture(source)
                with patch("automanip.cumulative_episodes."+constant, 1), patch("research.discovery.amp_v1.load_demos") as reader:
                    with self.assertRaises(ValueError):
                        import_legacy_demos(source, destination, TASK, INTERFACE)
                    reader.assert_not_called()
                self.assertFalse(destination.exists())

    def test_legacy_huge_declared_shape_preflight_and_source_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary)/"legacy", Path(temporary)/"episodes"
            legacy_fixture(source)
            with self.assertRaisesRegex(ValueError, "outside"):
                import_legacy_demos(source, source/"nested-import", TASK, INTERFACE)
            fake = io.BytesIO()
            np.lib.format.write_array_header_2_0(fake, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 51, 48)})
            raw = io.BytesIO()
            with zipfile.ZipFile(source/"trajectories.npz") as original, zipfile.ZipFile(raw, "w") as changed:
                for member in original.namelist():
                    changed.writestr(member, fake.getvalue() if member == "observations.npy" else original.read(member))
            (source/"trajectories.npz").write_bytes(raw.getvalue())
            with patch("research.discovery.amp_v1.load_demos") as reader:
                with self.assertRaisesRegex(ValueError, "shape"):
                    import_legacy_demos(source, destination, TASK, INTERFACE)
                reader.assert_not_called()
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
