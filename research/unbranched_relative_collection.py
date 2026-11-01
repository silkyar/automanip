"""Equal-total-physics control: independent episodes, no branch selection."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

import numpy as np

from research.prefix_search import proposal, raw_mean
from research.relative_gripper_search import relative_action


def run(inputs, output, index):
    import torch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import PolicySchema, load_policy
    from automanip.stackcube import STACK_TASK, check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter
    inputs, output = Path(inputs), Path(output); output.mkdir(exist_ok=False)
    plan = json.loads((inputs/'plan.json').read_text()); started = time.monotonic(); adapter = None
    record = dict(state='started', collection_index=index, transitions_reserved=0, transitions_completed=0,
                  optimizer_calls=0, policy_inference_rows=0, reset_calls=0, state_write_calls=0,
                  confirmation_queries=0, final_queries=0, batches=[], partition='training-only-unbranched-control')
    journal = (output/'work.jsonl').open('x', buffering=1)
    def persist():
        record['wall_seconds'] = time.monotonic()-started
        tmp = output/'report.json.tmp'; tmp.write_text(json.dumps(record, indent=2)+'\n'); tmp.replace(output/'report.json')
    try:
        if index not in (0, 1) or not plan['execution_authorized']: raise ValueError('Unregistered collection')
        record['versions'] = {k: version(k) for k in plan['production_pins']}
        if record['versions'] != plan['production_pins']: raise ValueError('Production pins differ')
        record['protocol_sha256'] = check_frozen_protocol()
        blob = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(blob.read_bytes()).hexdigest() != plan['physx_sha256']: raise ValueError('PhysX differs')
        record['gpu'] = torch.cuda.get_device_name(0)
        if record['gpu'] != 'NVIDIA L40S': raise ValueError('Hardware differs')
        for name, digest in plan['policy_files'].items():
            if hashlib.sha256((inputs/name).read_bytes()).hexdigest() != digest: raise ValueError('Parent artifact differs')
        parent = plan['parents'][index]; schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
        actor = NumpyPolicy(load_policy(inputs/plan['policies'][str(parent)], schema), schema)
        adapter = StackCubeAdapter(num_envs=64)
        if adapter.env.control_freq != 20 or adapter.env.sim_freq != 100: raise ValueError('Clocks differ')
        for batch in range(18):
            base = plan['control_episode_seed_bases'][index]+64*batch
            obs = np.asarray(adapter.reset(list(range(base, base+64))), dtype=np.float32); record['reset_calls'] += 1
            noise = np.random.default_rng(plan['control_noise_seed_bases'][index]+batch).standard_normal((50, 64, 8))
            perturbation = proposal(1, plan['control_proposal_seed_bases'][index]+batch*100)
            observations, actions, means, flags = [obs.copy()], [], [], []
            anchor = np.zeros(64)
            for step in range(50):
                mean = raw_mean(actor, obs); record['policy_inference_rows'] += 64
                if step == 30: anchor = obs[:, 7].copy()
                action = relative_action(mean, plan['gaussian_std'][str(parent)], noise[step], 1,
                                         perturbation, step, obs, anchor, 'current')
                if record['transitions_reserved']+64 > 57600: raise ValueError('Transition ceiling')
                record['transitions_reserved'] += 64
                journal.write(json.dumps(dict(event='reserved', count=record['transitions_reserved']))+'\n')
                result = adapter.env.step(torch.as_tensor(action, device=adapter.env.device))
                record['transitions_completed'] += 64
                journal.write(json.dumps(dict(event='completed', count=record['transitions_completed']))+'\n')
                obs = result[0].detach().cpu().numpy().copy(); signal = adapter.signals()
                success = signal['on_top'] & signal['static'] & ~signal['grasped']
                if not torch.equal(success, result[4]['success']): raise ValueError('Success parity differs')
                flags.append(torch.stack([signal[k] for k in ('grasped', 'lifted', 'transported', 'on_top', 'static', 'contact_free')]+[success], -1).cpu().numpy().astype(np.int8))
                observations.append(obs.copy()); actions.append(action); means.append(mean.astype(np.float32))
            outcome = np.asarray(flags)[-1, :, 6].astype(bool)
            np.savez_compressed(output/f'batch-{batch}.npz', observations=np.asarray(observations),
                                executed_actions=np.asarray(actions), parent_means=np.asarray(means), flags=np.asarray(flags),
                                perturbation=perturbation, outcome=outcome)
            record['batches'].append(dict(batch=batch, successes=int(outcome.sum()), episodes=64)); persist()
        if record['transitions_completed'] != 57600: raise ValueError('Incomplete collection')
        record['state'] = 'complete'
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        if adapter is not None: adapter.close()
        persist(); journal.close()
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', required=True); p.add_argument('--output', required=True); p.add_argument('--collection-index', type=int, required=True)
    a = p.parse_args(); r = run(a.inputs, a.output, a.collection_index)
    print(json.dumps({k: v for k, v in r.items() if k not in ('traceback', 'batches')}))
    raise SystemExit(0 if r['state'] == 'complete' else 1)
