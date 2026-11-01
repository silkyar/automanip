"""Fresh-process branching via original forward history; never sets state."""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

import numpy as np

from research.branching_validity import measure_frame
from research.prefix_search import ANCHOR, CANDIDATES, QUIET_SCALE, OFFSET_SCALE, candidate_kind, prefix_digest, proposal, raw_mean
from research.relative_gripper_search import CONTROLLED_FINGER, LOWER, UPPER
from research.onpolicy_support import action as support_action


def run(inputs, output, index, candidate, repeat, source_kind):
    import torch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import PolicySchema, load_policy
    from automanip.stackcube import STACK_TASK, check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter
    inputs, output = Path(inputs), Path(output)
    output.mkdir(exist_ok=False)
    plan = json.loads((inputs/'plan.json').read_text())
    started = time.monotonic(); adapter = None
    record = dict(state='started', collection_index=index, candidate=candidate, repeat=repeat,
                  kind='source_'+source_kind, source_kind=source_kind, transitions_reserved=0, transitions_completed=0,
                  vector_step_calls=0, reset_calls=0, state_write_calls=0, policy_inference_rows=0,
                  optimizer_calls=0, confirmation_queries=0, final_queries=0, partition='training-only', policy_noise_enabled=False)
    journal = (output/'work.jsonl').open('x', buffering=1)

    def persist():
        record['wall_seconds'] = time.monotonic()-started
        tmp = output/'report.json.tmp'
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n'); tmp.replace(output/'report.json')
    try:
        if index not in (0, 1) or repeat not in (0, 1) or not 0 <= candidate <= 3 or (repeat == 1 and candidate != 1) or source_kind not in ('original', 'student'):
            raise ValueError('Unregistered branch')
        if not plan['execution_authorized'] or (plan['anchor'], plan['bank_candidates'], plan['quiet_scale']) != (ANCHOR, 4, QUIET_SCALE):
            raise ValueError('Protocol differs')
        record['versions'] = {key: version(key) for key in plan['production_pins']}
        if record['versions'] != plan['production_pins']:
            raise ValueError('Production pins differ')
        for module, digest in plan['api_pins'].items():
            if hashlib.sha256(Path(importlib.import_module(module).__file__).read_bytes()).hexdigest() != digest:
                raise ValueError('API pin differs: '+module)
        record['protocol_sha256'] = check_frozen_protocol()
        blob = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(blob.read_bytes()).hexdigest() != plan['physx_sha256']:
            raise ValueError('PhysX pin differs')
        record['gpu'] = torch.cuda.get_device_name(0)
        if record['gpu'] != 'NVIDIA L40S':
            raise ValueError('Hardware differs')
        seed = plan['parents'][index]
        schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
        for name, digest in plan['policy_files'].items():
            if hashlib.sha256((inputs/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Parent artifact differs')
        actor = NumpyPolicy(load_policy(inputs/plan['source_policies'][str(seed)][source_kind], schema), schema)
        perturbation = proposal(candidate, plan['proposal_seed_bases'][index])
        adapter = StackCubeAdapter(num_envs=64)
        gripper = adapter.env.agent.controller.controllers['gripper']
        controlled = gripper.active_joint_indices[gripper.control_joint_indices].cpu().tolist()
        if controlled != [CONTROLLED_FINGER] or gripper.config.lower != LOWER or gripper.config.upper != UPPER or gripper.config.use_delta or not gripper.config.normalize_action:
            raise ValueError('Gripper mapping differs from analytical conversion')
        if adapter.env.control_freq != 20 or adapter.env.sim_freq != 100:
            raise ValueError('Effective clocks differ')
        first = plan['episode_seed_bases'][index]
        obs = np.asarray(adapter.reset(list(range(first, first+64))), dtype=np.float32)
        if not np.array_equal(obs[:, :9], adapter.env.agent.robot.get_qpos().cpu().numpy()):
            raise ValueError('Proprioception layout differs')
        record['reset_calls'] = 1
        anchor_finger = np.zeros(64)
        frames, observations, flags, actions, means = [measure_frame(adapter)], [obs.copy()], [], [], []
        for step in range(50):
            mean = raw_mean(actor, obs); record['policy_inference_rows'] += 64
            if step == ANCHOR:
                if not np.array_equal(obs[:, :9], adapter.env.agent.robot.get_qpos().cpu().numpy()):
                    raise ValueError('Anchor proprioception layout differs')
                anchor_finger = obs[:, CONTROLLED_FINGER].copy()
            action = support_action(mean, plan['gaussian_std'][str(seed)], perturbation,
                                    step, obs, anchor_finger, candidate)
            if record['transitions_reserved']+64 > plan['max_transitions_per_child']:
                raise ValueError('Transition cap')
            record['transitions_reserved'] += 64; record['vector_step_calls'] += 1
            journal.write(json.dumps(dict(event='reserved', count=record['transitions_reserved']))+'\n')
            result = adapter.env.step(torch.as_tensor(action, device=adapter.env.device))
            record['transitions_completed'] += 64
            journal.write(json.dumps(dict(event='completed', count=record['transitions_completed']))+'\n')
            obs = result[0].detach().cpu().numpy().copy()
            signal = adapter.signals()
            success = signal['on_top'] & signal['static'] & ~signal['grasped']
            if not torch.equal(success, result[4]['success']):
                raise ValueError('Independent success parity differs')
            flags.append(torch.stack([signal[k] for k in ('grasped', 'lifted', 'transported', 'on_top', 'static', 'contact_free')]+[success], -1).cpu().numpy().astype(np.int8))
            observations.append(obs.copy()); actions.append(action); means.append(mean.astype(np.float32))
            frames.append(measure_frame(adapter))
        trace = {**{k: np.stack([f[k] for f in frames]) for k in frames[0]},
                 'observations': np.stack(observations), 'flags': np.stack(flags),
                 'executed_actions': np.stack(actions), 'parent_means': np.stack(means)}
        record['prefix_sha256'] = prefix_digest(trace)
        record['final_successes'] = int(trace['flags'][-1, :, 6].sum())
        record['ever_successes'] = int(trace['flags'][:, :, 6].any(0).sum())
        np.savez_compressed(output/'trace.npz', **trace, perturbation=perturbation)
        if record['transitions_completed'] != plan['max_transitions_per_child']:
            raise ValueError('Incomplete candidate')
        record['state'] = 'complete'
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        if adapter is not None: adapter.close()
        persist(); journal.close()
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', required=True); p.add_argument('--output', required=True)
    p.add_argument('--collection-index', type=int, required=True); p.add_argument('--candidate', type=int, required=True); p.add_argument('--repeat', type=int, required=True); p.add_argument('--source-kind', choices=['original', 'student'], required=True)
    a = p.parse_args(); r = run(a.inputs, a.output, a.collection_index, a.candidate, a.repeat, a.source_kind)
    print(json.dumps({k: v for k, v in r.items() if k != 'traceback'}))
    raise SystemExit(0 if r['state'] == 'complete' else 1)
