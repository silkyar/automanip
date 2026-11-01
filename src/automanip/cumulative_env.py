"""Trusted, closed source/north variation adapters for the new track only.

Importing this module constructs no simulator. Real factories below are NOT run
by this milestone's tests. Historical task, scoring and artifact files are reused
unchanged. North is an initial-placement variation, not a new manipulation task.
"""
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from pathlib import Path

from automanip.cumulative import SOURCE, canonical
from automanip.evaluator import evaluate_batch
from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import load_policy
from automanip.stackcube import STACK_TASK, STACK_TASK_SOURCE_SHA256,check_frozen_protocol
from automanip.stackcube_adapter import StackCubeAdapter


def map_positions(a, b, goal):
    """Pure affine map of official initial samples; uses no additional RNG."""
    if goal not in ('source','north'):raise ValueError('Unregistered variation')
    left=a.clone() if hasattr(a,'clone') else a.copy()
    right=b.clone() if hasattr(b,'clone') else b.copy()
    if goal=='north':
        left[...,0]=.4*a[...,0];left[...,1]=.12+.1*a[...,1]
        right[...,0]=.25*b[...,0];right[...,1]=.34+b[...,1]/15
    return left,right


def _apply(owner, task):
    if task.goal=='source':return False
    from mani_skill.utils.structs.pose import Pose
    a,b=owner.cubeA.pose,owner.cubeB.pose
    left,right=map_positions(a.p,b.p,task.goal)
    owner.cubeA.set_pose(Pose.create_from_pq(p=left,q=a.q))
    owner.cubeB.set_pose(Pose.create_from_pq(p=right,q=b.q))
    # Actor.set_pose writes GPU staging, not simulator state. No physics step:
    # apply only rigid bodies, then refresh caches before returning observations.
    gpu=bool(getattr(owner,'gpu_sim_enabled',False))
    if gpu:
        owner.scene.px.gpu_apply_rigid_dynamic_data()
        owner.scene._gpu_fetch_all()
    return gpu


class GoalResetEnv:
    """Attach to the existing non-autoreset vector wrapper; no rollout loop."""
    def __init__(self, env, task):
        task.require_stack()
        self.base,self.task=env,task
        self.task_id,self.interface_id=task.task_id,task.interface.fingerprint
        self.goal_state_refresh_calls=0;self.goal_gpu_sync_calls=0
    def reset(self, **kwargs):
        observations,info=self.base.reset(**kwargs)
        if self.task.goal!='source':
            owner=self.base.unwrapped
            self.goal_gpu_sync_calls+=int(_apply(owner,self.task))
            observations=owner.get_obs()
            # Refresh after initial placement, never simulate counterfactuals.
            info=dict(info);info.update(owner.evaluate())
            self.goal_state_refresh_calls+=1
        return observations,info
    def step(self, actions):return self.base.step(actions)
    def close(self):return self.base.close()


def make_training_environment(task, *, num_envs=64, backend='physx_cuda'):
    """Future authorized-run entry point; deliberately never auto-invoked."""
    task.require_stack()
    if version('mani_skill')!=STACK_TASK.mani_skill_version:raise ValueError('Simulator pin differs')
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.envs.tasks.tabletop import stack_cube
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!=STACK_TASK_SOURCE_SHA256:raise ValueError('Task source differs')
    kwargs=STACK_TASK.env_kwargs(num_envs=num_envs,backend=backend);kwargs['reward_mode']='normalized_dense'
    owner=gym.make(STACK_TASK.env_id,**kwargs)
    wrapped=ManiSkillVectorEnv(owner,auto_reset=False,ignore_terminations=True,record_metrics=True)
    return GoalResetEnv(wrapped,task)


class VariationAdapter(StackCubeAdapter):
    """New trusted reset adapter; inherit historical physical success checks."""
    def __init__(self, task, num_envs=8, backend='physx_cuda'):
        task.require_stack();self.task=task;self.task_id=task.task_id;self.interface_id=task.interface.fingerprint;self.goal_state_refresh_calls=0;self.goal_gpu_sync_calls=0
        super().__init__(num_envs,backend)
    def reset(self, seeds):
        observations=super().reset(seeds)
        if self.task.goal!='source':
            self.goal_gpu_sync_calls+=int(_apply(self.env,self.task))
            observations=self.env.get_obs().detach().cpu().tolist()
            self.initial_height=self.env.cubeA.pose.p[:,2].clone()
            self.initial_distance=self.torch.linalg.vector_norm((self.env.cubeA.pose.p-self.env.cubeB.pose.p)[:,:2],dim=1)
            self.goal_state_refresh_calls+=1
        return observations


def assess_controller(policy_dir, task, seeds, *, partition, environment):
    """Reuse fixed-horizon evaluator with a trusted caller-owned adapter.

    Tests supply synthetic adapters only. A production supervisor must separately
    isolate this owner and enforce final-assessment admission. No trainer code is
    loaded from an artifact. Never import final trajectories into training.
    """
    try:
        task.require_stack()
        if partition not in ('development','final'):raise ValueError('Assessment partition required')
        protocol=check_frozen_protocol()
        artifact=load_policy(policy_dir,task.schema)
        if getattr(environment,'task_id',None)!=task.task_id or getattr(environment,'interface_id',None)!=task.interface.fingerprint:
            raise ValueError('Assessment environment/task identity differs')
        seeds=list(seeds)
        lanes=getattr(environment,'num_envs',len(seeds))
        if type(lanes) is not int or lanes<1 or not seeds or len(seeds)%lanes:
            raise ValueError('Assessment episodes must form complete fixed-size vector batches')
        results=[];policy=NumpyPolicy(artifact,task.schema)
        for start in range(0,len(seeds),lanes):
            results.extend(evaluate_batch(environment,policy,seeds[start:start+lanes],horizon=50,action_dim=8))
    finally:environment.close()
    return dict(historical_kernel_protocol_sha256=protocol,variation_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),task_id=task.task_id,weights_sha256=artifact.weights_sha256,partition=partition,
        episode_set_id=hashlib.sha256(canonical(list(seeds))).hexdigest(),episodes=len(results),
        successes=sum(r.outcome.success_at_end for r in results),failed_episodes=sum(r.outcome.failed for r in results),
        environment_transitions=sum(r.steps for r in results),outcomes=[asdict(r) for r in results],
        goal_state_refresh_calls=getattr(environment,'goal_state_refresh_calls',0),goal_gpu_sync_calls=getattr(environment,'goal_gpu_sync_calls',0))
