"""Generate a few motion-planned demonstrations as simulator-state trajectories (demo-assisted research track only).

Uses ManiSkill's own motion-planning solutions on CPU simulation, reset seeds 400,000,000+ (outside every
benchmark range). Records the full state dict after each step plus per-step success; actions are not kept,
because the planner uses pd_joint_pos while training uses pd_joint_delta_pos. Output is safe JSON, no pickle.
"""
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import ROOT, trainer_image

app = modal.App('automanip-demo-gen-20261101', include_source=False)
image = trainer_image.pip_install('mplib==0.1.1')
OUT = ROOT/'research/sample-efficiency-20261031/demos'
SEED_BASE = 400_000_000


@app.function(image=image, gpu='L4', cpu=4, memory=8192, timeout=1200, retries=0, include_source=False, serialized=True)
def generate(env_id: str, count: int, max_attempts: int = 40):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch
    module = {'PlaceSphere-v1': 'place_sphere', 'StackCube-v1': 'stack_cube', 'PlugCharger-v1': 'plug_charger'}[env_id]
    solve = __import__(f'mani_skill.examples.motionplanning.panda.solutions.{module}', fromlist=['solve']).solve

    def flat(state):
        return {group: {name: value[0].tolist() for name, value in items.items()} for group, items in state.items()}

    class Recorder(gym.Wrapper):
        def reset(self, **kwargs):
            out = self.env.reset(**kwargs)
            self.states, self.success = [flat(self.env.unwrapped.get_state_dict())], [False]
            return out

        def step(self, action):
            out = self.env.step(action)
            self.states.append(flat(self.env.unwrapped.get_state_dict()))
            self.success.append(bool(torch.as_tensor(out[-1]['success']).reshape(-1)[0]))
            return out

    env = Recorder(gym.make(env_id, obs_mode='none', control_mode='pd_joint_pos', render_mode=None,
                            sim_backend='physx_cpu', num_envs=1))
    demos, attempts = [], []
    for offset in range(max_attempts):
        seed = SEED_BASE + offset
        try:
            solve(env, seed=seed)
            ok = any(env.success)
        except Exception as exc:  # planner failures are recorded, not hidden
            attempts.append(dict(seed=seed, ok=False, error=type(exc).__name__)); continue
        attempts.append(dict(seed=seed, ok=ok, length=len(env.states)))
        if ok:
            first = env.success.index(True)
            demos.append(dict(seed=seed, states=env.states[:first+1], length=first+1))
        if len(demos) == count:
            break
    return dict(env_id=env_id, seed_base=SEED_BASE, demos=demos, attempts=attempts,
                environment_transitions=sum(a.get('length', 0) for a in attempts))


@app.local_entrypoint()
def main(env_id: str = 'PlaceSphere-v1', count: int = 5):
    result = generate.remote(env_id, count)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/f'{env_id}-{count}.json'
    path.write_text(json.dumps(result, allow_nan=False))
    print(json.dumps(dict(path=str(path), demos=[(d['seed'], d['length']) for d in result['demos']],
                          attempts=len(result['attempts']), transitions=result['environment_transitions'])))
