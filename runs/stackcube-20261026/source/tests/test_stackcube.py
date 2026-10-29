"""StackCube's frozen rules, cross-task separation and safe inference bridge."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from automanip.evaluator import EpisodeOutcome, evaluate_batch
from automanip.inference import MLPPolicy
from automanip.policy_io import PolicySchema, ValidatedPolicy, load_policy
from automanip.stackcube import (STACK_TASK, DEVELOPMENT_SEEDS, CONFIRMATION_SEEDS,
                                OFFICIAL_PPO_SHA256, StackFrame, StackDiagnostics, stackcube_success)
from automanip.task_spec import TASK


class StackCubeTests(unittest.TestCase):
    def test_task_is_separate_and_pickcube_unchanged(self):
        self.assertEqual(TASK.fingerprint,"aa34b8495725f90c42e151cfa42c59698bc89e02712cce302db253568661ea79")
        self.assertNotEqual(STACK_TASK.fingerprint,TASK.fingerprint)
        self.assertEqual((STACK_TASK.observation_dim,STACK_TASK.action_dim,STACK_TASK.horizon),(48,8,50))
        self.assertFalse(set(DEVELOPMENT_SEEDS)&set(CONFIRMATION_SEEDS))
        self.assertEqual((len(DEVELOPMENT_SEEDS),len(CONFIRMATION_SEEDS)),(128,512))

    def test_official_source_is_unmodified(self):
        source=Path(__file__).resolve().parents[1]/"third_party/maniskill/ppo.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),OFFICIAL_PPO_SHA256)

    def test_success_requires_geometry_release_and_static_cube(self):
        self.assertTrue(stackcube_success((0,0,.04),0,0,False))
        for offset,v,w,g in [((.04,0,.04),0,0,False),((0,0,.05),0,0,False),
                             ((0,0,.04),.011,0,False),((0,0,.04),0,.51,False),
                             ((0,0,.04),0,0,True)]:
            self.assertFalse(stackcube_success(offset,v,w,g))
        with self.assertRaises(ValueError):stackcube_success((0,0,float("nan")),0,0,False)

    def test_stability_is_consecutive_and_contact_free_is_diagnostic_only(self):
        tracker=StackDiagnostics()
        f=StackFrame(True,True,True,True,True,False)
        tracker.update(f)
        failure=EpisodeOutcome(False,False,False,True)
        self.assertEqual(tracker.report(failure)["first_unmet_stage"],"placed_still_grasped")
        f=replace(f,grasped=False)
        for _ in range(9):tracker.update(f)
        self.assertFalse(tracker.ever["stable_stack_10_steps"])
        tracker.update(replace(f,static=False))
        for _ in range(10):tracker.update(f)
        self.assertTrue(tracker.ever["stable_stack_10_steps"])
        self.assertFalse(tracker.ever["contact_free_placement"])
        self.assertTrue(f.success)  # stock non-grasping is distinct from zero contact
        self.assertEqual(tracker.report(failure)["first_unmet_stage"],"lost_stack_before_horizon")
        self.assertEqual(tracker.report(replace(failure,failed=True))["first_unmet_stage"],"execution_failure")

    def test_existing_rollout_supports_frozen_eight_action_task(self):
        class Env:
            def reset(self,seeds):return [[0.0]*48 for _ in seeds]
            def step(self,actions):return [[0.0]*48],[True],[False],[False]
        class Policy:
            def act(self,obs):return [[0.0]*8]
        result,=evaluate_batch(Env(),Policy(),[1],horizon=2,action_dim=8)
        self.assertTrue(result.outcome.success_at_end)
        result,=evaluate_batch(Env(),Policy(),[1],horizon=2)
        self.assertTrue(result.outcome.failed)  # PickCube still rejects eight actions

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"optional NumPy")
    def test_vectorized_policy_matches_reference_and_rejects_invalid_inputs(self):
        import numpy as np
        from automanip.numpy_inference import NumpyPolicy
        schema=PolicySchema(STACK_TASK.fingerprint,48,8)
        rng=np.random.default_rng(22)
        tensors={}
        for i,(left,right) in enumerate(zip((48,16,16),(16,16,8))):
            tensors[f"layers.{i}.weight"]=tuple(rng.normal(0,.1,right*left))
            tensors[f"layers.{i}.bias"]=tuple(rng.normal(0,.1,right))
        artifact=ValidatedPolicy((16,16),tensors,"0"*64)
        obs=rng.normal(size=(8,48)).tolist()
        np.testing.assert_allclose(NumpyPolicy(artifact,schema).act(obs),MLPPolicy(artifact,schema).act(obs),rtol=1e-12,atol=1e-12)
        with self.assertRaises(ValueError):NumpyPolicy(artifact,schema).act([[float("nan")]*48])

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional PyTorch")
    def test_official_checkpoint_conversion_and_cross_task_rejection(self):
        import torch
        from scripts.export_official_policy import convert
        torch.manual_seed(55)
        state={"actor_logstd":torch.zeros(1,8)}
        for prefix,nout in (("actor_mean",8),("critic",1)):
            dims=(48,256,256,256,nout)
            for i,(left,right) in enumerate(zip(dims,dims[1:])):
                state[f"{prefix}.{2*i}.weight"]=torch.randn(right,left)*.01
                state[f"{prefix}.{2*i}.bias"]=torch.zeros(right)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/"official.pt"
            torch.save(state,checkpoint)
            record=convert(checkpoint,Path(tmp)/"policy")
            artifact=load_policy(Path(tmp)/"policy",PolicySchema(STACK_TASK.fingerprint,48,8))
            self.assertEqual(record["weights_sha256"],artifact.weights_sha256)
            with self.assertRaises(ValueError):load_policy(Path(tmp)/"policy",PolicySchema(TASK.fingerprint,42))
            state["actor_mean.0.weight"]=torch.zeros(256,42)
            torch.save(state,checkpoint)
            with self.assertRaises(ValueError):convert(checkpoint,Path(tmp)/"bad")


if __name__=="__main__":unittest.main()
