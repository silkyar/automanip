"""Synthetic CPU-only integration tests. No deployment or simulator imports."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from research.discovery.amp_v1 import StudentActor,export_actor
from automanip.cumulative import (SOURCE,NORTH,STACK_INTERFACE,TaskContract,register_asset,
    load_asset,asset_status,compatible,cost,lifetime_cost,retention_decision)
from automanip.cumulative_ppo import AdaptationAgent,Budget,adapt,default_config,state_digest
from automanip.cumulative_env import map_positions,assess_controller
from automanip.cumulative_episodes import load_episodes
from automanip.numpy_inference import NumpyPolicy


class FakeEnv:
    """An in-memory algebraic fixture, NOT a robot/simulator."""
    def __init__(self,task=NORTH,n=2,invalid=False):
        self.task_id=task.task_id;self.interface_id=task.interface.fingerprint
        self.n=n;self.steps=0;self.reset_calls=0;self.closed=False;self.invalid=invalid
    def reset(self,seed):
        self.reset_calls+=1;return torch.zeros(self.n,48),{}
    def step(self,action):
        self.steps+=1
        return torch.full((self.n,48),self.steps/100),torch.full((self.n,),float('nan') if self.invalid else 1.),torch.zeros(self.n,dtype=torch.bool),torch.zeros(self.n,dtype=torch.bool),{}
    def close(self):self.closed=True


class CumulativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        torch.manual_seed(13);self.actor=StudentActor((8,));export_actor(self.actor,self.root/'source-policy')
        self.source=register_asset(self.root/'source.json',self.root/'source-policy',SOURCE,cost(10,2.,None),root=self.root)
        self.config=replace(default_config(7),num_envs=2,num_steps=2,total_timesteps=8,
                            hidden_sizes=(8,),update_epochs=1,num_minibatches=1,max_seconds=10.)
    def test_compatibility_is_semantic_not_vector_padding(self):
        self.assertNotEqual(SOURCE.task_id,NORTH.task_id)
        self.assertEqual(SOURCE.interface.fingerprint,NORTH.interface.fingerprint)
        compatible(self.source,NORTH)
        for change in ({'controller':'pd_ee_delta_pose'},{'robot':'panda'},
                       {'observation_semantics':'different frame'},{'action_semantics':'different units'},
                       {'control_hz':40},{'observation_dim':42},{'action_dim':7}):
            bad=TaskContract('north',replace(STACK_INTERFACE,**change))
            with self.subTest(change=change),self.assertRaises(ValueError):compatible(self.source,bad)
    def test_warmstart_actions_equal_source_and_fresh_state_is_explicit(self):
        torch.manual_seed(42);agent=AdaptationAgent(self.config,self.source.policy)
        torch.manual_seed(42);fresh=AdaptationAgent(self.config)
        self.assertEqual(state_digest(agent.critic),state_digest(fresh.critic))
        self.assertEqual(agent.logstd.tolist(),[[-.5]*8])
        obs=torch.linspace(-1,1,48).repeat(3,1)
        with torch.no_grad():actual=agent.actor.deterministic(obs)
        expected=torch.tensor(NumpyPolicy(self.source.policy,SOURCE.schema).act(obs.tolist()))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        with self.assertRaises(ValueError):AdaptationAgent(replace(self.config,hidden_sizes=(16,)),self.source.policy)
    def test_missing_tampered_and_immutable_assets(self):
        self.assertEqual(asset_status(self.root/'absent.json',root=self.root)['state'],'missing')
        with self.assertRaises(FileExistsError):register_asset(self.root/'source.json',self.root/'source-policy',SOURCE,cost(),root=self.root)
        p=self.root/'source.json';record=json.loads(p.read_bytes());record['acquisition']['environment_transitions']=0;p.write_text(json.dumps(record))
        with self.assertRaises(ValueError):load_asset(p,root=self.root)
    def test_real_ppo_update_exports_child_and_preserves_parent(self):
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.policy_dir.iterdir()}
        env=FakeEnv();record,child=adapt(NORTH,self.source,self.config,Budget(4,4,10),env,self.root/'attempt',root=self.root)
        self.assertEqual(record['status'],'training_interaction_ceiling',record)
        self.assertEqual(record['environment_transitions_confirmed'],4)
        self.assertEqual(record['optimizer_calls_completed'],1)
        self.assertEqual(child.record['parents'][0]['asset_id'],self.source.asset_id)
        self.assertEqual(child.task.task_id,NORTH.task_id)
        self.assertEqual(record['retention'].split(';')[0],'unassessed')
        self.assertFalse(record['initialization']['resume_supported'])
        self.assertEqual(record['initialization']['optimizer_initial_state_entries'],0)
        self.assertTrue(env.closed)
        episodes=load_episodes(self.root/'attempt/episodes/shard-00000',NORTH.task_id,NORTH.interface.fingerprint,for_training=True)
        self.assertEqual(sum(len(x['actions']) for x in episodes),4)
        self.assertTrue(all(not x['metadata']['complete'] for x in episodes))
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.policy_dir.iterdir()}
        self.assertEqual(before,after)
        self.assertEqual(child.record['acquisition']['environment_transitions'],4)
        self.assertIsNone(lifetime_cost(child,root=self.root)['totals']['dollars'])
        self.assertEqual(lifetime_cost(child,root=self.root)['totals']['environment_transitions'],14)
    def test_partial_update_budget_is_accounted(self):
        cfg=replace(self.config,update_epochs=3)
        record,child=adapt(NORTH,self.source,cfg,Budget(8,1,10),FakeEnv(),self.root/'limited',root=self.root)
        self.assertEqual(record['status'],'censored');self.assertEqual(record['optimizer_calls_attempted'],1)
        self.assertEqual(record['optimizer_calls_completed'],1);self.assertEqual(record['environment_transitions_confirmed'],4)
        self.assertIsNotNone(child)
    def test_zero_time_and_invalid_rollout_fail_without_fabricated_asset(self):
        env=FakeEnv();record,child=adapt(NORTH,self.source,self.config,Budget(4,1,0),env,self.root/'zero',root=self.root)
        self.assertEqual(record['status'],'censored');self.assertEqual(env.steps,0);self.assertIsNone(child)
        env=FakeEnv(invalid=True);record,child=adapt(NORTH,self.source,self.config,Budget(4,1,10),env,self.root/'bad',root=self.root)
        self.assertEqual(record['status'],'failed');self.assertEqual(record['environment_transitions_attempted'],2)
        self.assertEqual(record['optimizer_calls_completed'],0);self.assertIsNone(child)
    def test_invalid_actor_rejected_before_step(self):
        def invalid(agent,obs,actions=None):return torch.full((len(obs),8),float('nan')),torch.zeros(len(obs)),torch.zeros(len(obs)),torch.zeros(len(obs))
        env=FakeEnv()
        with patch.object(AdaptationAgent,'action_and_value',invalid):
            record,child=adapt(NORTH,self.source,self.config,Budget(4,1,10),env,self.root/'nan',root=self.root)
        self.assertEqual(record['status'],'failed');self.assertEqual(env.steps,0);self.assertIsNone(child)
    def test_final_feedback_cannot_guide_updates_and_failed_cost_stays_charged(self):
        def final_feedback(directory,step):return [dict(partition='final',weights_sha256='a'*64,environment_transitions=100)]
        record,child=adapt(NORTH,self.source,self.config,Budget(4,1,10,200),FakeEnv(),self.root/'feedback',root=self.root,feedback=final_feedback,feedback_reserve=200)
        self.assertEqual(record['status'],'failed');self.assertEqual(record['development_interactions_charged'],200)
        self.assertEqual(record['optimizer_calls_completed'],0);self.assertIsNone(child)
    def test_parent_cost_is_counted_once_through_shared_ancestry(self):
        b=register_asset(self.root/'b.json',self.root/'source-policy',NORTH,cost(3,1.,0),root=self.root,parents=(self.source,))
        c=register_asset(self.root/'c.json',self.root/'source-policy',NORTH,cost(4,1.,0),root=self.root,parents=(self.source,b))
        accounting=lifetime_cost(c,root=self.root)
        self.assertEqual(accounting['unique_assets'],3);self.assertEqual(accounting['totals']['environment_transitions'],17)
    def test_goal_mapping_preserves_height_and_changes_goal_support(self):
        a=np.array([[-.2,-.3,.02],[.2,.3,.02]]);b=a.copy();old=a.copy()
        aa,bb=map_positions(a,b,'north')
        np.testing.assert_array_equal(a,old)
        np.testing.assert_allclose(aa[:,1],[.09,.15]);np.testing.assert_allclose(bb[:,1],[.32,.36])
        np.testing.assert_array_equal(aa[:,2],a[:,2]);self.assertTrue((bb[:,1]>.3).all())
    def test_same_controller_retention_not_archived_router(self):
        def report(task,success,weights='b'*64):return dict(task_id=task.task_id,partition='development',episodes=128,successes=success,weights_sha256=weights,episode_set_id='paired')
        target=report(NORTH,120);retained=report(SOURCE,118);reference=report(SOURCE,124,'a'*64)
        decision=retention_decision(target,retained,reference,controller_sha256='b'*64,reference_sha256='a'*64)
        self.assertTrue(decision['promoted'])
        with self.assertRaises(ValueError):retention_decision(target,reference,reference,controller_sha256='b'*64,reference_sha256='a'*64)
        retained['successes']=117
        self.assertFalse(retention_decision(target,retained,reference,controller_sha256='b'*64,reference_sha256='a'*64)['promoted'])
    def test_existing_evaluator_kernel_is_used_with_synthetic_adapter_only(self):
        class Adapter:
            task_id=NORTH.task_id;interface_id=NORTH.interface.fingerprint;closed=False
            def reset(self,seeds):return [[0.]*48 for _ in seeds]
            def step(self,actions):return [[0.]*48 for _ in actions],[True]*len(actions),[False]*len(actions),[False]*len(actions)
            def close(self):self.closed=True
        env=Adapter();report=assess_controller(self.source.policy_dir,NORTH,[11,12],partition='development',environment=env)
        self.assertEqual(report['successes'],2);self.assertEqual(report['environment_transitions'],100)
        self.assertTrue(env.closed)
        # This checks orchestration on algebraic fixtures, never robot performance.

    def test_assessment_splits_fixed_vector_batches_without_padding(self):
        class Adapter:
            task_id=NORTH.task_id;interface_id=NORTH.interface.fingerprint;num_envs=2;closed=False
            def __init__(self):self.batches=[]
            def reset(self,seeds):
                if len(seeds)!=self.num_envs:raise ValueError('seed count mismatch')
                self.batches.append(list(seeds));return [[0.]*48 for _ in seeds]
            def step(self,actions):return [[0.]*48 for _ in actions],[True]*len(actions),[False]*len(actions),[False]*len(actions)
            def close(self):self.closed=True
        env=Adapter();report=assess_controller(self.source.policy_dir,NORTH,[1,2,3,4,5,6],partition='development',environment=env)
        self.assertEqual(env.batches,[[1,2],[3,4],[5,6]])
        self.assertEqual(report['successes'],6);self.assertEqual(report['failed_episodes'],0)
        self.assertEqual(report['environment_transitions'],300);self.assertTrue(env.closed)
        env=Adapter()
        with self.assertRaises(ValueError):assess_controller(self.source.policy_dir,NORTH,[1,2,3],partition='development',environment=env)
        self.assertTrue(env.closed);self.assertEqual(env.batches,[])

    def test_no_extra_reset_after_last_allowed_transition(self):
        class DoneEnv(FakeEnv):
            def step(self,action):
                obs,reward,term,trunc,info=super().step(action)
                return obs,reward,term,torch.ones(self.n,dtype=torch.bool),info
        env=DoneEnv();cfg=replace(self.config,num_steps=1)
        record,child=adapt(NORTH,self.source,cfg,Budget(4,4,10),env,self.root/'done',root=self.root)
        self.assertEqual(record['status'],'training_interaction_ceiling',record)
        self.assertEqual(env.reset_calls,2);self.assertEqual(record['explicit_reset_calls'],2)
        self.assertEqual(record['explicit_reset_attempts'],2)
    def test_feedback_overage_records_actual_consumption_before_failure(self):
        def feedback(directory,step):
            weights=json.loads((directory/'manifest.json').read_bytes())['weights_sha256']
            return [dict(partition='development',weights_sha256=weights,environment_transitions=250)]
        record,child=adapt(NORTH,self.source,self.config,Budget(4,1,10,200),FakeEnv(),self.root/'over',root=self.root,feedback=feedback,feedback_reserve=200)
        self.assertEqual(record['status'],'failed');self.assertEqual(record['development_interactions_confirmed'],250)
        self.assertEqual(record['development_interactions_charged'],250);self.assertIsNone(child)
    def test_owned_environments_closed_when_preflight_or_artifact_fails(self):
        env=FakeEnv(SOURCE)
        with self.assertRaises(ValueError):adapt(NORTH,self.source,self.config,Budget(4,1,10),env,self.root/'reject',root=self.root)
        self.assertTrue(env.closed);self.assertEqual(env.steps,0)
        env=FakeEnv()
        with self.assertRaises(FileNotFoundError):assess_controller(self.root/'missing-policy',NORTH,[1],partition='development',environment=env)
        self.assertTrue(env.closed)
    def test_paired_retention_counts_must_match(self):
        target=dict(task_id=NORTH.task_id,partition='development',episodes=128,successes=120,weights_sha256='b'*64,episode_set_id='paired')
        retained=dict(target,task_id=SOURCE.task_id);reference=dict(retained,episodes=256,successes=240,weights_sha256='a'*64)
        with self.assertRaises(ValueError):retention_decision(target,retained,reference,controller_sha256='b'*64,reference_sha256='a'*64)
    def test_interrupted_optimizer_is_reserved_and_not_claimed_complete(self):
        original=torch.optim.Adam.step
        def interrupted(optimizer,*args,**kwargs):
            original(optimizer,*args,**kwargs)
            raise RuntimeError('synthetic interruption after parameter mutation')
        with patch.object(torch.optim.Adam,'step',interrupted):
            record,child=adapt(NORTH,self.source,self.config,Budget(4,2,10),FakeEnv(),self.root/'interrupted',root=self.root)
        self.assertEqual(record['status'],'failed');self.assertIsNone(child)
        self.assertEqual(record['optimizer_calls_attempted'],1);self.assertEqual(record['optimizer_calls_completed'],0)
        self.assertEqual(record['optimizer_calls_uncertain'],1)
        events=[json.loads(x)['event'] for x in (self.root/'interrupted/work.jsonl').read_text().splitlines()]
        self.assertIn('optimizer_reserved',events);self.assertNotIn('optimizer_completed',events)
        from automanip.cumulative_ppo import recover_work
        recovered=recover_work(self.root/'interrupted/work.jsonl')
        self.assertEqual(recovered['optimizer_calls_uncertain'],1)
        self.assertEqual(recovered['environment_completed'],4)
    def test_gpu_pose_api_order_is_checked_without_a_gpu(self):
        from automanip.cumulative_env import _apply
        events=[]
        class Pose:
            @staticmethod
            def create_from_pq(p,q):return SimpleNamespace(p=p,q=q)
        def actor(name):
            return SimpleNamespace(pose=SimpleNamespace(p=np.zeros((2,3)),q=np.tile([1.,0.,0.,0.],(2,1))),set_pose=lambda value:events.append(name))
        owner=SimpleNamespace(cubeA=actor('A'),cubeB=actor('B'),gpu_sim_enabled=True,
            scene=SimpleNamespace(px=SimpleNamespace(gpu_apply_rigid_dynamic_data=lambda:events.append('apply')),_gpu_fetch_all=lambda:events.append('fetch')))
        modules={name:ModuleType(name) for name in ('mani_skill','mani_skill.utils','mani_skill.utils.structs','mani_skill.utils.structs.pose')}
        modules['mani_skill.utils.structs.pose'].Pose=Pose
        with patch.dict(sys.modules,modules):self.assertTrue(_apply(owner,NORTH))
        self.assertEqual(events,['A','B','apply','fetch'])
    def test_fresh_actor_uses_existing_ppo_initialization_scheme(self):
        torch.manual_seed(42);agent=AdaptationAgent(self.config)
        self.assertAlmostEqual(float(agent.actor.net[-1].weight.detach().norm()),.04,places=5)
        self.assertTrue(torch.equal(agent.actor.net[-1].bias,torch.zeros(8)))
    def test_exact_plan_compiler_is_data_only_and_rejects_mutations(self):
        from automanip.cumulative_plan import make_plan,compile_plan
        actor=StudentActor((256,256,256));export_actor(actor,self.root/'large-policy')
        source=register_asset(self.root/'large.json',self.root/'large-policy',SOURCE,cost(10,2.,None),root=self.root)
        plan=make_plan(source,root=self.root)
        result=compile_plan(plan,self.root/'prepared.json',root=self.root)
        self.assertEqual(result['state'],'prepared_not_executed');self.assertFalse(result['ready_for_robot_execution'])
        self.assertEqual(len(result['arms']),3)
        changed=json.loads(json.dumps(plan));changed['budget_per_trainable_arm']['seconds']=601
        with self.assertRaises(ValueError):compile_plan(changed,self.root/'bad-prepared.json',root=self.root)
        changed['execution_authorized']=True
        with self.assertRaises(PermissionError):compile_plan(changed,self.root/'run.json',root=self.root)

    def test_existing_output_and_bad_assessment_partition_close_owned_env(self):
        output=self.root/'already';output.mkdir();env=FakeEnv()
        with self.assertRaises(FileExistsError):adapt(NORTH,self.source,self.config,Budget(4,1,10),env,output,root=self.root)
        self.assertTrue(env.closed);self.assertEqual(env.steps,0)
        env=FakeEnv()
        with self.assertRaises(ValueError):assess_controller(self.source.policy_dir,NORTH,[1],partition='training',environment=env)
        self.assertTrue(env.closed)

    def test_development_feedback_costs_are_not_training_replay(self):
        def feedback(directory,step):
            weights=json.loads((directory/'manifest.json').read_bytes())['weights_sha256']
            return [dict(task_id=task.task_id,partition='development',weights_sha256=weights,environment_transitions=100) for task in (SOURCE,NORTH)]
        record,child=adapt(NORTH,self.source,self.config,Budget(4,2,10,200),FakeEnv(),self.root/'valid-feedback',root=self.root,feedback=feedback,feedback_at=(0,),feedback_reserve=200)
        self.assertEqual(record['status'],'training_interaction_ceiling',record)
        self.assertEqual(record['development_interactions_charged'],200)
        self.assertEqual(record['development_interactions_confirmed'],200)
        self.assertEqual(child.record['acquisition']['environment_transitions'],204)
        from automanip.cumulative_ppo import recover_work
        self.assertEqual(recover_work(self.root/'valid-feedback/work.jsonl')['development_interactions_charged'],200)
        episodes=load_episodes(self.root/'valid-feedback/episodes/shard-00000',NORTH.task_id,NORTH.interface.fingerprint,for_training=True)
        self.assertEqual(sum(len(x['actions']) for x in episodes),4)


if __name__=='__main__':unittest.main()
