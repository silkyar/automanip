"""Only the two original unqueried primary scores; no trainer or new endpoint."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app, ROOT
from evaluate import frozen_evaluate
from transport import pack_weights


@app.local_entrypoint()
def reconcile():
    import time
    import modal
    sys.path.insert(0, str(ROOT/'src'))
    from automanip.policy_io import PolicySchema, load_policy

    root = ROOT/'runs/joint-exploration-20261029'
    plan_raw = (root/'source/plan.json').read_bytes()
    plan = json.loads(plan_raw)
    original = json.loads((root/'trusted-summary.json').read_bytes())
    if hashlib.sha256(plan_raw).hexdigest() != original['plan_sha256'] or not plan['admitted']:
        raise PermissionError('Submitted plan differs')
    proof = json.loads((ROOT/'.runtime/joint-exploration-archive-verification.json').read_bytes())
    if proof['state'] != 'verified': raise PermissionError('Archive not verified')
    budget = json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd']) > 100:
        raise PermissionError('Shared cap exceeded')
    if time.time()-budget['observed_at'] > 300: raise PermissionError('Budget ledger stale')
    if not any(row['campaign'] == 'joint-exploration-20261029' and float(row['reserved_cap_usd']) == 2 for row in budget['reservations']):
        raise PermissionError('Original campaign reservation missing')
    schema = PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f', 48, 8)
    payloads = []
    for route in ('joint', 'coordinate'):
        arm = original['arms'][route]
        if arm['state'] != 'failed' or 'UnicodeDecodeError' not in arm.get('error', '') or 'development' in arm:
            raise PermissionError('Only the recorded pre-score receipt decoding failures are admitted')
        receipt = json.loads((root/route/'receipt.json').read_bytes())
        report = receipt['report']
        if receipt['state'] != 'complete' or report['state'] != 'complete' or report['counts']['transitions'] != 102400:
            raise ValueError('No completed original primary')
        if receipt['route'] != route or receipt['plan_sha256'] != original['plan_sha256']:
            raise ValueError('Receipt route or plan differs')
        expected = {name: plan['source_sha256'][name] for name in ('replay_robot.py', 'replay_learner.py', 'sac_baseline.py', 'joint_exploration.py')}
        if report['source_files_sha256'] != expected: raise ValueError('Original core source differs')
        if report['behavior_mode'] != route or report['behavior_epsilon'] != .1:
            raise ValueError('Behavior intervention differs')
        admission = report['admission']
        if (admission['route'] != route or admission['collector_sha256'] != plan['source_sha256']['joint_exploration.py']
            or admission['launcher_sha256'] != plan['source_sha256']['joint_exploration_train.py']):
            raise ValueError('Collector/launcher provenance differs')
        initial = load_policy(root/route/'run/actor-step-0', schema)
        if initial.weights_sha256 != plan['initial_actor_sha256']: raise ValueError('Initial safe actor differs')
        point = next(row for row in report['checkpoint_exports'] if row['label'] == 'actor-step-102400' and row['transitions'] == 102400)
        policy = root/route/'run/actor-step-102400'
        actor = load_policy(policy, schema)
        if actor.weights_sha256 != point['weights_sha256']: raise ValueError('Original final actor differs')
        payloads.append((route, policy, actor.weights_sha256))
    launch = ROOT/'.runtime/modal-joint-exploration-score-handoff-launch.json'
    if launch.exists(): raise RuntimeError('Handoff already dispatched; read existing calls/results, no repeat')
    record = dict(state='started', plan_sha256=original['plan_sha256'], arms={},
                  new_training_runs=0, new_confirmation_queries=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with launch.open('x') as stream: json.dump(record, stream, indent=2)
    volume = modal.Volume.from_name('automanip-modal-validation')
    evaluator = frozen_evaluate.with_options(gpu='L40S', timeout=120, max_containers=1)

    def existing(path):
        raw = bytearray()
        try:
            for block in volume.read_file(path):
                raw.extend(block)
                if len(raw) > 4*1024*1024: raise ValueError('Bounded score read exceeded')
        except FileNotFoundError:
            return None
        # Unreadable or incomplete existing attempts are never treated as absent.
        return json.loads(raw)

    for route, policy, digest in payloads:
        path = 'frozen-evaluation-attempts/fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764/'+digest+'/development/result.json'
        score = existing(path)
        arm = dict(weights_sha256=digest, durable_report_path=path, reused_existing_report=score is not None)
        record['arms'][route] = arm
        if score is not None:
            if score['state'] != 'complete' or score['weights_sha256'] != digest:
                raise RuntimeError('Existing GPU attempt not complete; no repeat')
            arm['state'] = 'reused'
        else:
            arm.update(state='submitting', submitted_at=time.time())
            launch.write_text(json.dumps(record, indent=2)+'\n')
            call = evaluator.spawn((policy/'manifest.json').read_bytes(),
                pack_weights((policy/'weights.json').read_bytes()), 'development', 'L40S', time.time(), 100)
            arm.update(state='submitted', function_call_id=call.object_id)
            launch.write_text(json.dumps(record, indent=2)+'\n')
            # Keep the local entrypoint connected; each original GPU call ID is
            # durable before its own wait. No redispatch.
            result = call.get(timeout=160)
            arm.update(state='complete', result=result)
        launch.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(dict(route=route, state=arm['state'], weights_sha256=digest)))
    record['state'] = 'complete'
    launch.write_text(json.dumps(record, indent=2)+'\n')
