"""Does writing the current state back perturb its future? Training diagnostic."""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

import numpy as np

from research.branching_validity import capture, measure_frame, restore


CONDITIONS = ('no_write', 'physical', 'augmented')


def run(inputs, output, condition, repeat):
    import torch
    from automanip.stackcube import check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter
    inputs, output = Path(inputs), Path(output)
    output.mkdir(exist_ok=False)
    plan = json.loads((inputs/'plan.json').read_text())
    record = dict(state='started', condition=condition, repeat=repeat,
                  transitions_reserved=0, transitions_completed=0,
                  vector_step_calls=0, captures=0, restores=0, reset_calls=0,
                  policy_inference_rows=0, optimizer_calls=0,
                  confirmation_queries=0, final_queries=0)
    journal = (output/'work.jsonl').open('x', buffering=1)
    started = time.monotonic(); adapter = None

    def persist():
        record['wall_seconds'] = time.monotonic()-started
        temporary = output/'report.json.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'report.json')
    try:
        if condition not in CONDITIONS or repeat not in (0, 1) or not plan['execution_authorized']:
            raise ValueError('Unregistered condition')
        record['versions'] = {key: version(key) for key in plan['production_pins']}
        if record['versions'] != plan['production_pins']:
            raise ValueError('Production versions differ')
        for module, digest in plan['api_pins'].items():
            if hashlib.sha256(Path(importlib.import_module(module).__file__).read_bytes()).hexdigest() != digest:
                raise ValueError('Pinned API differs')
        record['protocol_sha256'] = check_frozen_protocol()
        blob = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(blob.read_bytes()).hexdigest() != plan['physx_sha256']:
            raise ValueError('PhysX differs')
        record['gpu'] = torch.cuda.get_device_name(0)
        if record['gpu'] != 'NVIDIA L40S':
            raise ValueError('Hardware differs')
        path = inputs/'action-tape.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan['action_tape_sha256']:
            raise ValueError('Input tape drift')
        with np.load(path, allow_pickle=False) as saved:
            actions = saved['actions'].copy()
        if actions.shape != (50, 64, 8) or actions.dtype != np.float32 or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
            raise ValueError('Invalid tape')
        adapter = StackCubeAdapter(num_envs=64)
        adapter.reset(list(range(plan['episode_seed_base'], plan['episode_seed_base']+64)))
        record['reset_calls'] += 1
        frames, flags = [measure_frame(adapter)], []
        for t, action in enumerate(actions):
            if t == plan['anchor_step']:
                snapshot = capture(adapter.env); record['captures'] += 1
                if condition != 'no_write':
                    restore(adapter.env, snapshot, augmented=condition == 'augmented')
                    record['restores'] += 1
            if record['transitions_reserved']+64 > plan['max_transitions_per_child']:
                raise ValueError('Transition cap')
            record['transitions_reserved'] += 64; record['vector_step_calls'] += 1
            journal.write(json.dumps(dict(event='reserved', count=record['transitions_reserved']))+'\n')
            result = adapter.env.step(torch.as_tensor(action, device=adapter.env.device))
            record['transitions_completed'] += 64
            journal.write(json.dumps(dict(event='completed', count=record['transitions_completed']))+'\n')
            signal = adapter.signals()
            success = signal['on_top'] & signal['static'] & ~signal['grasped']
            if not torch.equal(success, result[4]['success']):
                raise ValueError('Success parity differs')
            flags.append(torch.stack([signal[k] for k in ('grasped', 'lifted', 'transported', 'on_top', 'static', 'contact_free')]+[success], -1).cpu().numpy().astype(np.int8))
            frames.append(measure_frame(adapter))
        np.savez_compressed(output/'trace.npz', **{k: np.stack([f[k] for f in frames]) for k in frames[0]}, flags=np.stack(flags))
        if record['transitions_completed'] != plan['max_transitions_per_child']:
            raise ValueError('Incomplete diagnostic')
        record['state'] = 'complete'
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        if adapter is not None: adapter.close()
        persist(); journal.close()
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', required=True); p.add_argument('--output', required=True)
    p.add_argument('--condition', choices=CONDITIONS, required=True); p.add_argument('--repeat', type=int, required=True)
    args = p.parse_args(); r = run(args.inputs, args.output, args.condition, args.repeat)
    print(json.dumps({k: v for k, v in r.items() if k != 'traceback'}))
    raise SystemExit(0 if r['state'] == 'complete' else 1)
