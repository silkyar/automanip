"""Pinned ManiSkill adapter. Optional imports happen only during construction."""
from importlib.metadata import PackageNotFoundError, version
from automanip.metrics import pickcube_success
from automanip.policy_io import PolicySchema
from automanip.task_spec import TASK


class ManiSkillAdapter:
    def __init__(self, num_envs=1, backend="physx_cpu"):
        try:
            installed_version = version("mani_skill")
        except PackageNotFoundError as exc:
            raise RuntimeError("Install the optional pinned simulator environment before running this command") from exc
        if installed_version != TASK.mani_skill_version:
            raise RuntimeError("ManiSkill version differs from benchmark pin")
        import gymnasium as gym
        import mani_skill.envs  # registers tasks
        import torch
        self.torch = torch
        self.owner = gym.make(TASK.env_id, **TASK.env_kwargs(num_envs=num_envs, backend=backend))
        # Deliberately bypass wrappers: BaseEnv never autoresets and we own horizon.
        self.env = self.owner.unwrapped
        self.num_envs = num_envs
        try:
            action = self.env.single_action_space
            obs = self.env.single_observation_space
            if action.shape != (7,) or not (action.low == -1).all() or not (action.high == 1).all():
                raise ValueError("unexpected action space")
            if obs.shape != (TASK.observation_dim,):
                raise ValueError("expected flattened state observations")
            self.schema = PolicySchema(TASK.fingerprint, int(obs.shape[0]))
        except Exception:
            self.owner.close()
            raise

    def reset(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError("seed batch mismatch")
        obs, _ = self.env.reset(seed=list(seeds))
        return obs.detach().cpu().tolist()

    def step(self, actions):
        tensor = self.torch.tensor(actions, dtype=self.torch.float32, device=self.env.device)
        obs, _, terminated, truncated, _ = self.env.step(tensor)
        # Score from simulator state; never accept candidate-provided reward/info.
        distance = self.torch.linalg.vector_norm(self.env.goal_site.pose.p-self.env.cube.pose.p, dim=1)
        velocities = self.env.agent.robot.get_qvel()[..., :7].detach().cpu().tolist()
        success = [pickcube_success(d, tuple(v)) for d, v in zip(distance.detach().cpu().tolist(), velocities)]
        return (obs.detach().cpu().tolist(), success,
                terminated.detach().cpu().tolist(), truncated.detach().cpu().tolist())

    def close(self):
        self.owner.close()
