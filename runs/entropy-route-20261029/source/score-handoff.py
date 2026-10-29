"""Score only the archived primary missed by a file-visibility handoff error."""
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

    root = ROOT/'runs/entropy-route-20261029'
    plan_raw = (root/'source/plan.json').read_bytes()
    plan = json.loads(plan_raw)
    original = json.loads((root/'trusted-summary.json').read_bytes())
    if hashlib.sha256(plan_raw).hexdigest() != original['plan_sha256'] or not plan['admitted']:
        raise PermissionError('Submitted plan differs')
    route = 'high-actor-low-backup'
    arm = original['arms'][route]
    if arm['state'] != 'failed' or 'FileNotFoundError' not in arm.get('error', '') or 'development' in arm:
        raise PermissionError('Only the recorded missing-receipt handoff is admitted')
    proof = json.loads((ROOT/'.runtime/entropy-route-archive-verification.json').read_bytes())
    if proof['state'] != 'verified': raise PermissionError('Archive not verified')
    receipt = json.loads((root/route/'receipt.json').read_bytes())
    report = receipt['report']
    if receipt['route'] != route or receipt['plan_sha256'] != original['plan_sha256'] or report['admission']['route'] != route:
        raise ValueError('Archived receipt route or submitted plan differs')
    if receipt['state'] != 'complete' or report['state'] != 'complete' or report['counts']['transitions'] != 102400:
        raise ValueError('No completed primary checkpoint')
    expected = {name: plan['source_sha256'][name] for name in ('replay_robot.py', 'replay_learner.py', 'sac_baseline.py')}
    if report['source_files_sha256'] != expected: raise ValueError('Original core source differs')
    if report['admission']['base_learner_sha256'] != plan['source_sha256']['replay_learner_base.py']:
        raise ValueError('Original base learner differs')
    schema = PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f', 48, 8)
    initial = load_policy(root/route/'run/actor-step-0', schema)
    if initial.weights_sha256 != plan['initial_actor_sha256']: raise ValueError('Initial safe policy differs')
    point = next(row for row in report['checkpoint_exports'] if row['label'] == 'actor-step-102400' and row['transitions'] == 102400)
    policy = root/route/'run/actor-step-102400'
    actor = load_policy(policy, schema)
    if actor.weights_sha256 != point['weights_sha256']: raise ValueError('Primary safe policy differs')
    budget = json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd']) > 100:
        raise PermissionError('Shared cap exceeded')
    if time.time()-budget['observed_at'] > 300: raise PermissionError('Budget ledger stale')
    if not any(row['campaign'] == 'entropy-route-20261029' and float(row['reserved_cap_usd']) == 2 for row in budget['reservations']):
        raise PermissionError('Original reservation missing')
    durable_path = 'frozen-evaluation-attempts/fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764/'+actor.weights_sha256+'/development/result.json'
    volume = modal.Volume.from_name('automanip-modal-validation')
    try:
        raw = bytearray()
        for block in volume.read_file(durable_path):
            raw.extend(block)
            if len(raw) > 4*1024*1024: raise ValueError('Score read bound')
    except FileNotFoundError:
        raw = None
    launch_path = ROOT/'.runtime/modal-entropy-route-score-handoff-launch.json'
    if launch_path.exists(): raise RuntimeError('Handoff already dispatched; read existing result, no repeat')
    record = dict(route=route, weights_sha256=actor.weights_sha256, plan_sha256=original['plan_sha256'],
                  submitted_at=time.time(), reused_existing_report=raw is not None,
                  durable_report_path=durable_path, new_training_runs=0, new_confirmation_queries=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if raw is not None:
        score = json.loads(raw)
        if score['state'] != 'complete' or score['weights_sha256'] != actor.weights_sha256:
            raise RuntimeError('Existing evaluator attempt is not complete; no repeat')
        record['state'] = 'reused'
    else:
        record['state'] = 'submitting'
        with launch_path.open('x') as stream: json.dump(record, stream, indent=2)
        evaluator = frozen_evaluate.with_options(gpu='L40S', timeout=120, max_containers=1)
        call = evaluator.spawn((policy/'manifest.json').read_bytes(),
                               pack_weights((policy/'weights.json').read_bytes()), 'development', 'L40S', time.time(), 100)
        record.update(state='submitted', function_call_id=call.object_id)
    launch_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))
