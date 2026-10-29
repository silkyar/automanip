"""Evaluator-owned StackCube state scoring, independent of trainer reward/info."""
import hashlib
from importlib.metadata import version
from pathlib import Path

from automanip.policy_io import PolicySchema
from automanip.stackcube import STACK_TASK, STACK_TASK_SOURCE_SHA256, StackFrame, StackDiagnostics


class StackCubeAdapter:
    def __init__(self, num_envs=8, backend="physx_cuda"):
        if version("mani_skill") != STACK_TASK.mani_skill_version:
            raise RuntimeError("ManiSkill version differs from benchmark")
        import torch
        import gymnasium as gym
        import mani_skill.envs
        from mani_skill.envs.tasks.tabletop import stack_cube
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest() != STACK_TASK_SOURCE_SHA256:
            raise RuntimeError("official StackCube task source changed")
        self.torch = torch
        self.num_envs = num_envs
        self.owner = gym.make(STACK_TASK.env_id, reconfiguration_freq=1,
                              **STACK_TASK.env_kwargs(num_envs=num_envs,backend=backend))
        self.env = self.owner.unwrapped
        try:
            if self.env.single_observation_space.shape != (48,) or self.env.single_action_space.shape != (8,):
                raise ValueError("unexpected StackCube spaces")
            action = self.env.single_action_space
            if not (action.low == -1).all() or not (action.high == 1).all():
                raise ValueError("unexpected StackCube action bounds")
            self.schema = PolicySchema(STACK_TASK.fingerprint,48,8)
        except Exception:
            self.owner.close()
            raise

    def reset(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError("seed count mismatch")
        self.diagnostics = [StackDiagnostics() for _ in seeds]
        obs,_ = self.env.reset(seed=list(seeds))
        self.initial_height = self.env.cubeA.pose.p[:,2].clone()
        self.initial_distance = self.torch.linalg.vector_norm((self.env.cubeA.pose.p-self.env.cubeB.pose.p)[:,:2],dim=1)
        self.parity_mismatches = 0
        return obs.detach().cpu().tolist()

    def signals(self):
        t = self.torch
        a,b = self.env.cubeA,self.env.cubeB
        offset = a.pose.p-b.pose.p
        half = self.env.cube_half_size
        on_top = (t.linalg.vector_norm(offset[:,:2],dim=1) <= t.linalg.vector_norm(half[:2])+0.005) & ((offset[:,2]-half[2]*2).abs() <= 0.005)
        static = (t.linalg.vector_norm(a.linear_velocity,dim=1) <= 0.01) & (t.linalg.vector_norm(a.angular_velocity,dim=1) <= 0.5)
        grasped = self.env.agent.is_grasping(a)
        lifted = a.pose.p[:,2] >= self.initial_height+0.03
        transported = lifted & ((self.initial_distance-t.linalg.vector_norm(offset[:,:2],dim=1)) >= 0.03)
        left = self.env.scene.get_pairwise_contact_forces(self.env.agent.finger1_link,a)
        right = self.env.scene.get_pairwise_contact_forces(self.env.agent.finger2_link,a)
        contact_free = (t.linalg.vector_norm(left,dim=1) <= 1e-6) & (t.linalg.vector_norm(right,dim=1) <= 1e-6)
        return dict(grasped=grasped,lifted=lifted,transported=transported,on_top=on_top,
                    static=static,contact_free=contact_free)

    def step(self, actions):
        obs,_,terminated,truncated,_ = self.env.step(self.torch.tensor(actions,dtype=self.torch.float32,device=self.env.device))
        signals = {k:v.detach().cpu().tolist() for k,v in self.signals().items()}
        frames = [StackFrame(**{k:v[i] for k,v in signals.items()}) for i in range(self.num_envs)]
        successes = [f.success for f in frames]
        for tracker,frame in zip(self.diagnostics,frames):
            tracker.update(frame)
        return obs.detach().cpu().tolist(),successes,terminated.detach().cpu().tolist(),truncated.detach().cpu().tolist()

    def close(self):
        self.owner.close()
