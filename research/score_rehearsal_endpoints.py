"""Independent public scoring: safe MLPs only, no learner data or proposal code."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback


def run(inputs, output, index, method):
    import torch
    from automanip.evaluator import evaluate_batch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import PolicySchema, load_policy
    from automanip.stackcube import STACK_TASK, check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter
    inputs, output = Path(inputs), Path(output); output.mkdir(exist_ok=False)
    plan = json.loads((inputs/'plan.json').read_text()); started = time.monotonic(); adapter = None
    record = dict(state='started', collection_index=index, method=method, environment_reserved=0,
                  environment_completed=0, optimizer_calls=0, confirmation_queries=0, final_queries=0,
                  partition='fresh-public-policy-only', batches=[], episodes=[])
    journal = (output/'work.jsonl').open('x', buffering=1)
    def persist():
        record['wall_seconds'] = time.monotonic()-started
        tmp = output/'report.json.tmp'; tmp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n'); tmp.replace(output/'report.json')
    try:
        if index not in (0, 1) or method not in ('base', 'rehearsal', 'teacher_only') or not plan['execution_authorized']:
            raise ValueError('Unregistered assessment')
        record['versions'] = {k: version(k) for k in plan['production_pins']}
        if record['versions'] != plan['production_pins']: raise ValueError('Version pin differs')
        record['gpu'] = torch.cuda.get_device_name(0)
        if record['gpu'] != 'NVIDIA L40S': raise ValueError('Hardware differs')
        record['protocol_sha256'] = check_frozen_protocol()
        blob = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(blob.read_bytes()).hexdigest() != plan['physx_sha256']: raise ValueError('PhysX differs')
        parent = plan['parents'][index]
        path = inputs/'policies'/str(parent)/method
        for filename in ('manifest.json', 'weights.json'):
            relative = str(path.relative_to(inputs)/filename)
            if hashlib.sha256((path/filename).read_bytes()).hexdigest() != plan['policy_files'][relative]:
                raise ValueError('Safe policy drift')
        actor = load_policy(path, PolicySchema(STACK_TASK.fingerprint, 48, 8))
        record.update(parent=parent, weights_sha256=actor.weights_sha256)
        adapter = StackCubeAdapter(64)
        class CountedPhysics:
            def __init__(self, base): self.base = base
            def __getattr__(self, name): return getattr(self.base, name)
            def step(self, actions):
                if record['environment_reserved']+64 > 12800: raise ValueError('Scoring transition ceiling')
                record['environment_reserved'] += 64
                journal.write(json.dumps(dict(event='reserved', count=record['environment_reserved']))+'\n')
                result = self.base.step(actions)
                record['environment_completed'] += 64
                journal.write(json.dumps(dict(event='completed', count=record['environment_completed']))+'\n')
                return result
        adapter.env = CountedPhysics(adapter.env)
        policy = NumpyPolicy(actor, adapter.schema)
        for batch in range(4):
            base = plan['evaluation_seed_bases'][index]+64*batch
            results = evaluate_batch(adapter, policy, list(range(base, base+64)), horizon=50, action_dim=8)
            episodes = [dict(**asdict(r), diagnostics=d.report(r.outcome)) for r, d in zip(results, adapter.diagnostics)]
            record['episodes'].extend(episodes)
            record['batches'].append(dict(batch=batch, successes=sum(r.outcome.success_at_end for r in results),
                                           ever_grasp=sum(e['diagnostics']['ever']['grasp'] for e in episodes),
                                           failures=sum(r.outcome.failed for r in results)))
            persist()
            if any(r.outcome.failed for r in results): raise ValueError('Scorer execution failure; episodes retained')
        if record['environment_completed'] != 12800: raise ValueError('Incomplete scoring')
        record.update(state='complete', successes=sum(b['successes'] for b in record['batches']),
                      ever_grasp=sum(b['ever_grasp'] for b in record['batches']), episode_count=256)
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        if adapter is not None: adapter.close()
        persist(); journal.close()
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', required=True); p.add_argument('--output', required=True)
    p.add_argument('--collection-index', type=int, required=True); p.add_argument('--method', required=True)
    a = p.parse_args(); r = run(a.inputs, a.output, a.collection_index, a.method)
    print(json.dumps({k: v for k, v in r.items() if k not in ('traceback', 'episodes')}))
    raise SystemExit(0 if r['state'] == 'complete' else 1)
