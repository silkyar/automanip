"""Bounded restoration diagnostic; no optimizer, planning, or private scoring."""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

import numpy as np

from research.branching_validity import TOLERANCES, capture, compare, restore


def run(inputs, output):
    import torch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import PolicySchema, load_policy
    from automanip.stackcube import STACK_TASK, check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter

    inputs, output = Path(inputs), Path(output)
    output.mkdir(exist_ok=False)
    plan = json.loads((inputs/'plan.json').read_text())
    started = time.monotonic()
    record = dict(state='started', transitions_reserved=0, transitions_completed=0,
                  vector_step_calls=0, reset_calls=0, captures=0, restores=0,
                  policy_inference_rows=0, optimizer_calls=0,
                  protected_queries=0, confirmation_queries=0, final_queries=0,
                  comparisons=[], partition='training-only-branching-diagnostic')
    journal = (output/'work.jsonl').open('x', buffering=1)
    adapter = None

    def persist():
        record['wall_seconds'] = time.monotonic()-started
        temporary = output/'report.json.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'report.json')

    def step(action):
        n = plan['num_envs']
        if record['transitions_reserved']+n > plan['max_transitions']:
            raise ValueError('Transition ceiling')
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (n, 8) or not np.isfinite(action).all() or np.abs(action).max() > 1:
            raise ValueError('Invalid action tape')
        record['transitions_reserved'] += n
        record['vector_step_calls'] += 1
        journal.write(json.dumps(dict(event='reserved', count=record['transitions_reserved']))+'\n')
        result = adapter.env.step(torch.as_tensor(action, device=adapter.env.device))
        record['transitions_completed'] += n
        journal.write(json.dumps(dict(event='completed', count=record['transitions_completed']))+'\n')
        flags = adapter.signals()
        independent = flags['on_top'] & flags['static'] & ~flags['grasped']
        if not torch.equal(independent, result[4]['success']):
            raise ValueError('Success parity mismatch')
        packed = torch.stack([flags[k] for k in
                              ('grasped', 'lifted', 'transported', 'on_top', 'static', 'contact_free')]
                             +[independent], dim=-1).cpu().numpy().astype(np.int8)
        return result[0].detach().cpu().numpy().copy(), packed

    def frame():
        e = adapter.env
        cpu = lambda t: t.detach().cpu().numpy().copy()
        return dict(position=cpu(torch.stack([e.cubeA.pose.p, e.cubeB.pose.p, e.agent.tcp.pose.p], 1)),
                    quaternion=cpu(torch.stack([e.cubeA.pose.q, e.cubeB.pose.q, e.agent.tcp.pose.q], 1)),
                    qpos=cpu(e.agent.robot.get_qpos()), qvel=cpu(e.agent.robot.get_qvel()),
                    linear_velocity=cpu(torch.stack([e.cubeA.linear_velocity, e.cubeB.linear_velocity], 1)),
                    angular_velocity=cpu(torch.stack([e.cubeA.angular_velocity, e.cubeB.angular_velocity], 1)))

    def pack(frames, flags):
        return {**{key: np.stack([f[key] for f in frames]) for key in frames[0]},
                'flags': np.stack(flags)}

    try:
        if plan['execution_authorized'] is not True or plan['tolerances'] != TOLERANCES:
            raise ValueError('Unregistered diagnostic')
        record['versions'] = {key: version(key) for key in plan['production_pins']}
        if record['versions'] != plan['production_pins']:
            raise ValueError('Production versions differ')
        for module, digest in plan['api_pins'].items():
            path = Path(importlib.import_module(module).__file__)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Pinned API differs: '+module)
        for name, digest in plan['policy_files'].items():
            if hashlib.sha256((inputs/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Safe actor differs')
        record['protocol_sha256'] = check_frozen_protocol()
        physx = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(physx.read_bytes()).hexdigest() != plan['physx_sha256']:
            raise ValueError('PhysX differs')
        record['gpu'] = torch.cuda.get_device_name(0)
        if record['gpu'] != 'NVIDIA L40S':
            raise ValueError('Hardware differs')
        adapter = StackCubeAdapter(num_envs=plan['num_envs'])
        if adapter.env.control_freq != 20 or adapter.env.sim_freq != 100:
            raise ValueError('Clock differs')
        record['controllers'] = {name: dict(use_target=c.config.use_target,
                                          interpolate=c.config.interpolate)
                                 for name, c in adapter.env.agent.controller.controllers.items()}
        for parent_index, seed in enumerate(plan['parents']):
            record['reset_calls'] += 1
            obs = np.asarray(adapter.reset(list(range(plan['episode_seed_bases'][parent_index],
                                  plan['episode_seed_bases'][parent_index]+plan['num_envs']))), dtype=np.float32)
            policy = NumpyPolicy(load_policy(inputs/plan['policies'][str(seed)],
                                            PolicySchema(STACK_TASK.fingerprint, 48, 8)), adapter.schema)
            rng = np.random.default_rng(plan['noise_seed_bases'][parent_index])
            frames, flags, actions, observations, snapshots = [frame()], [], [], [obs.copy()], {}
            for t in range(50):
                if t in plan['anchor_steps']:
                    snapshots[t] = capture(adapter.env)
                    record['captures'] += 1
                mean = np.asarray(policy.act(obs))
                record['policy_inference_rows'] += plan['num_envs']
                action = np.clip(mean+rng.standard_normal(mean.shape)*plan['gaussian_std'][str(seed)], -1, 1).astype(np.float32)
                obs, f = step(action)
                actions.append(action); observations.append(obs.copy()); flags.append(f); frames.append(frame())
            reference = pack(frames, flags)
            np.savez_compressed(output/f'parent-{seed}-reference.npz', **reference,
                                actions=np.stack(actions), observations=np.stack(observations))
            record.setdefault('parent_stage_counts', {})[str(seed)] = dict(
                zip(('grasp', 'lift', 'transport', 'on_top', 'static', 'contact_free', 'success'),
                    np.any(reference['flags'], axis=0).sum(0).tolist()))
            # Alternate restore variants; use the same ordered anchor schedule in
            # every repeat. No new actions are selected from restored observations.
            for repeat in range(plan['repeats']):
                for anchor in plan['anchor_steps']:
                    order = ('physical', 'augmented') if repeat % 2 == 0 else ('augmented', 'physical')
                    for mode in order:
                        restore(adapter.env, snapshots[anchor], augmented=mode == 'augmented')
                        record['restores'] += 1
                        initial = frame()
                        replay_frames, replay_flags = [initial], []
                        for action in actions[anchor:]:
                            _, f = step(action)
                            replay_frames.append(frame()); replay_flags.append(f)
                        replay = pack(replay_frames, replay_flags)
                        ref = {key: value[anchor:] for key, value in reference.items()}
                        comparison = compare(ref, replay)
                        comparison.update(parent=seed, repeat=repeat, anchor=anchor, mode=mode)
                        # Separate immediate restoration from subsequent dynamics.
                        comparison['initial_errors'] = {
                            key: float(np.abs(reference[key][anchor]-initial[key]).max())
                            for key in TOLERANCES if key != 'rotation'}
                        record['comparisons'].append(comparison)
                        np.savez_compressed(output/f'parent-{seed}-{mode}-a{anchor}-r{repeat}.npz', **replay)
                        persist()
            # No snapshot, native checkpoint or untrusted pickle is persisted.
        if record['transitions_completed'] != plan['max_transitions']:
            raise ValueError('Diagnostic accounting differs')
        record['gate_passed'] = {mode: all(r['passed'] for r in record['comparisons'] if r['mode'] == mode)
                                 for mode in ('physical', 'augmented')}
        record['state'] = 'complete'
        record['decision'] = ('Branching validated on this finite diagnostic only; planning requires its own registration'
                              if record['gate_passed']['augmented'] else
                              'Stop before distillation; investigate exposed-state versus hidden-solver mismatch')
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        if adapter is not None:
            adapter.close()
        persist(); journal.close()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.inputs, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'traceback')}))
    raise SystemExit(0 if result['state'] == 'complete' else 1)
