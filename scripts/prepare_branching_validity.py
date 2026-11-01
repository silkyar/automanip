"""Prepare one bounded branching diagnostic, using already acquired safe actors."""
import hashlib
import json
from pathlib import Path
import shutil
import time

from research.branching_validity import TOLERANCES


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'.runtime/branching-validity-20261030'


def main():
    previous = ROOT/'.runtime/continuation-correction-20261030/inputs'
    old = json.loads((previous/'plan.json').read_text())
    if (DEST/'inputs/plan.json').exists():
        raise FileExistsError('Do not overwrite a registered attempt')
    inputs = DEST/'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for seed in (2, 3):
        if not (inputs/'policies'/str(seed)).exists():
            shutil.copytree(previous/'policies'/str(seed), inputs/'policies'/str(seed))
    from importlib.metadata import distribution
    api_modules = ('mani_skill.envs.sapien_env', 'mani_skill.envs.scene',
                   'mani_skill.agents.base_agent', 'mani_skill.agents.controllers.base_controller',
                   'mani_skill.agents.controllers.pd_joint_pos', 'mani_skill.utils.structs.articulation')
    pins = {name: hashlib.sha256(Path(distribution('mani_skill').locate_file(name.replace('.', '/')+'.py')).read_bytes()).hexdigest()
            for name in api_modules}
    plan = dict(format='automanip-branching-validity-v1', created_at=time.time(),
                execution_authorized=True,
                user_authorization='User approved branching diagnostic then bounded planning/distillation test; autonomous research and Modal GPU authorized',
                production_pins=old['production_pins'], api_pins=pins,
                physx_sha256='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706',
                parents=[2, 3], policies=old['policies'],
                policy_files={k: v for k, v in old['policy_files'].items() if k.startswith('policies/')},
                source_prefixes=old['source_prefixes'], gaussian_std=old['gaussian_std'],
                num_envs=64, horizon=50, anchor_steps=[0, 15, 30, 40], repeats=3,
                episode_seed_bases=[95900000, 95901000], noise_seed_bases=[95910000, 95911000],
                modes=['physical', 'augmented'], tolerances=TOLERANCES,
                max_transitions=94720, optimizer_calls=0, confirmation_queries=0, final_queries=0,
                partition='training-only diagnostic; never private or confirmation seeds',
                hypothesis='Exposed controller/drive/clock/RNG restoration may explain physical replay divergence; hidden PhysX solver state remains unobserved',
                primary='All augmented replays must match original uninterrupted futures within preregistered physical tolerances, with identical grasp/success flags at every control step. No tolerance relaxation after seeing results.',
                accounting='2 parents x 64 lanes x (50 original + 3 repeats x 2 restore modes x (50+35+20+10) replay steps)=94720 transitions; zero updates. Resets/capture/restore/extra state reads separately counted.',
                limitation='Finite diagnostic, not an exact-cloning theorem, learned-policy benchmark, or from-scratch efficiency result. Contacts/solver internal state cannot be captured by these APIs.',
                followup='If augmented gate fails, stop before distillation and diagnose mismatch. If it passes, register Plan-to-Distill with total new physics <=1M including this diagnostic and equal-cost rollout-learning control.')
    for name, digest in plan['policy_files'].items():
        if hashlib.sha256((inputs/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Parent artifact drift')
    raw = (json.dumps(plan, indent=2)+'\n').encode()
    (inputs/'plan.json').write_bytes(raw)
    (ROOT/'cumulative/experiments/branching-validity-v1.json').write_bytes(raw)
    print(json.dumps(dict(prepared=str(DEST), max_transitions=plan['max_transitions'])))


if __name__ == '__main__':
    main()
