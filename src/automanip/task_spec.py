"""Evaluator-owned interface; changes require a new benchmark identity."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class TaskSpec:
    env_id: str = "PickCube-v1"
    robot: str = "panda"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    horizon: int = 50
    mani_skill_version: str = "3.0.0b21"
    # Measured with the pinned ManiSkill CPU task on macOS.
    observation_dim: int = 42
    action_dim: int = 7

    @property
    def fingerprint(self) -> str:
        data = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode()).hexdigest()

    def env_kwargs(self, *, num_envs: int = 1, backend: str = "physx_cpu") -> dict:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if backend not in {"physx_cpu", "physx_cuda"}:
            raise ValueError("unsupported backend")
        if backend == "physx_cpu" and num_envs != 1:
            raise ValueError("ManiSkill CPU simulation requires num_envs=1")
        return dict(robot_uids=self.robot, obs_mode=self.obs_mode,
                    control_mode=self.control_mode, max_episode_steps=self.horizon,
                    num_envs=num_envs, sim_backend=backend, reward_mode="none")


TASK = TaskSpec()
