"""Meaningful SAC/replay/public-export invariants; no simulator imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"optional existing torch environment required")
class SACContractTests(unittest.TestCase):
    def test_metadata_allows_sanitized_image_without_git(self):
        from research.discovery.sac_baseline import Config,metadata
        with mock.patch("subprocess.check_output",side_effect=FileNotFoundError("git absent")):
            result=metadata(Config())
        self.assertIsNone(result["git_sha"])
        self.assertEqual(result["observation_dim"],48)

    def test_truncation_keeps_final_observation_and_bootstrap(self):
        from research.discovery.sac_baseline import transition_successor
        reset=torch.tensor([[99.],[88.],[33.]])
        term=torch.tensor([False,True,False]); trunc=torch.tensor([True,False,False])
        actual,stop=transition_successor(reset,term,trunc,{
            "final_observation":torch.tensor([[7.],[8.],[-100.]]),
            "_final_observation":term|trunc})
        self.assertEqual(actual.flatten().tolist(),[7.,8.,33.])
        self.assertEqual(stop.tolist(),[False,True,False])
        self.assertEqual(reset.flatten().tolist(),[99.,88.,33.])
        with self.assertRaises(ValueError): transition_successor(reset,term,trunc,{})

    def test_replay_wraparound_preserves_joint_transitions(self):
        from research.discovery.sac_baseline import Replay
        replay=Replay(3,1,1)
        for pair in ([1.,2.],[3.,4.]):
            x=torch.tensor(pair).reshape(-1,1)
            replay.add(x,x+10,x+20,x.flatten()+30,torch.zeros(2))
        self.assertEqual(replay.size,3)
        self.assertEqual(set(replay.storage["obs"].flatten().tolist()),{2.,3.,4.})
        sample=replay.sample(100)
        self.assertTrue(torch.equal(sample["next_obs"],sample["obs"]+10))
        self.assertTrue(torch.equal(sample["action"],sample["obs"]+20))
        self.assertTrue(torch.equal(sample["reward"],sample["obs"]+30))

    def test_actual_boolean_terminal_result_enters_float_replay(self):
        from research.discovery.sac_baseline import Replay,transition_successor
        reset=torch.tensor([[99.],[88.],[33.]])
        term=torch.tensor([False,True,False]);trunc=torch.tensor([True,False,False])
        actual,stop=transition_successor(reset,term,trunc,{
            "final_observation":torch.tensor([[7.],[8.],[-100.]]),
            "_final_observation":term|trunc})
        self.assertEqual(stop.dtype,torch.bool)
        replay=Replay(3,1,1)
        replay.add(torch.zeros(3,1),actual,torch.zeros(3,1),torch.ones(3),stop)
        self.assertEqual(replay.storage["stop"].dtype,torch.float32)
        self.assertEqual(replay.storage["stop"].flatten().tolist(),[0.,1.,0.])
        self.assertEqual(replay.storage["next_obs"].flatten().tolist(),[7.,8.,33.])

    def test_true_terminal_target_and_polyak_updates(self):
        from research.discovery.sac_baseline import SAC
        torch.manual_seed(1)
        sac=SAC(obs_dim=3,action_dim=2,width=16,tau=.1)
        old=[p.clone() for p in sac.t1.parameters()]
        batch=dict(obs=torch.randn(8,3),next_obs=torch.randn(8,3),action=torch.randn(8,2),
                   reward=torch.full((8,1),2.),stop=torch.ones(8,1))
        report=sac.update(batch)
        self.assertEqual(report["target_mean"],2.)
        for before,now,target in zip(old,sac.q1.parameters(),sac.t1.parameters()):
            self.assertTrue(torch.allclose(target,.9*before+.1*now,atol=1e-7))

    def test_squashed_gaussian_extreme_jacobian_gradients_remain_finite(self):
        from research.discovery.sac_baseline import Actor
        actor=Actor(obs_dim=3,action_dim=2,width=16)
        with torch.no_grad(): actor.mean.bias.fill_(100.)
        action,logprob=actor.sample(torch.ones(8,3))
        self.assertTrue(torch.isfinite(logprob).all())
        self.assertTrue((action.abs()<=1).all())
        logprob.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))

    def test_public_safe_export_matches_actor_and_hash(self):
        from research.discovery.sac_baseline import Actor,export_actor,numerical_action
        torch.manual_seed(9)
        actor=Actor(width=256)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"actor"
            exported=export_actor(actor,root)
            raw=(root/"weights.json").read_bytes()
            manifest=json.loads((root/"manifest.json").read_text())
            self.assertEqual(hashlib.sha256(raw).hexdigest(),manifest["weights_sha256"])
            self.assertEqual(manifest["hidden_sizes"],[256,256,8])
            self.assertEqual(set(manifest),{"format","task_fingerprint","observation_dim","action_dim","hidden_sizes","activation","weights_sha256"})
            states=torch.cat((torch.randn(128,48),torch.full((1,48),1000.),torch.full((1,48),-1000.)))
            with torch.no_grad():
                error=(actor.deterministic(states)-numerical_action(states,json.loads(raw))).abs().max()
            self.assertLess(float(error),1e-6)
            self.assertLess(exported["bytes"],32*1024*1024)

    def test_budget_and_vector_collection_are_bounded(self):
        from research.discovery.sac_baseline import Config,training_deadline
        self.assertEqual(Config().validate().max_seconds,1320)
        with self.assertRaises(ValueError): Config(max_seconds=1321).validate()
        with self.assertRaises(ValueError): Config(num_envs=128,training_freq=64).validate()
        with self.assertRaises(ValueError): Config(total_transitions=31).validate()
        self.assertEqual(training_deadline(Config(),100.,200.),200.)
        self.assertEqual(training_deadline(Config(),100.,2000.),1400.)
        with self.assertRaises(TimeoutError):training_deadline(Config(),100.,99.)


if __name__=="__main__":unittest.main()
