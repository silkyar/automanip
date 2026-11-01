"""Collect the Oct 31-Nov 1 sample-efficiency runs into one compact record, never run policies.

StackCube batches: trainer receipts from the watch logs plus frozen public-development score reports.
Transfer batches: trainer receipts with research-side evaluations (seeds 300,000,000+, ManiSkill success).
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_sample_efficiency import sustained_threshold  # noqa: E402

STACKCUBE = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'r1', 'r2', 'o1', 'r3', 'o2', 'r4', 'o3']
TRANSFER = ['t1', 't2', 't3', 't4', 't5', 't6', 'r5', 'r6', 'd1', 'a1']
OUT = ROOT/'docs/sample-efficiency-runs-20261031.json'


def watch(batch):
    path = ROOT/f'.runtime/se-watch-automanip-sample-efficiency-{batch}-20261031-scores.jsonl'
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not any(e['event'] == 'done' for e in events):
        return None
    return path, {e['index']: e['result'] for e in events if e['event'] == 'trained'}


def row(batch, receipt, source):
    assert receipt['state'] == 'complete' and receipt['protected_queries'] == 0
    assert receipt['training_initialization'] == 'from_scratch'
    config = receipt['config']
    return dict(batch=batch, arm=receipt['arm'], seed=receipt['seed'], task=config.get('env_id', 'StackCube-v1'),
                algo=config.get('algo', 'ppo'), config=config, gpu=receipt['gpu'],
                trainer_seconds=receipt['trainer_process_seconds'],
                environment_transitions=receipt['actual_work']['environment_completed'],
                optimizer_calls=receipt['actual_work']['optimizer_completed'], source=source)


rows, sources = [], {}
for batch in STACKCUBE:
    path, receipts = watch(batch)
    for index, receipt in sorted(receipts.items()):
        report_path = ROOT/f'.runtime/sample-efficiency-{batch}-20261031/dl/scores/trial-{index}/report.json'
        raw = report_path.read_bytes(); sources[str(report_path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        report = json.loads(raw)
        assert report['state'] == 'complete' and (report['arm'], report['seed']) == (receipt['arm'], receipt['seed'])
        assert report['confirmation_queries'] == 0 and report['partition'] == 'public-development'
        points = [(p['steps'], p['successes']) for p in report['points']]
        r = row(batch, receipt, str(path.relative_to(ROOT)))
        r.update(evaluation='frozen-public-development', public_seed_base=report['public_seed_base'],
                 final_successes=points[-1][1], final_weights_sha256=report['points'][-1]['weights_sha256'],
                 curve=points, T80=sustained_threshold(points, 103), T90=sustained_threshold(points, 116))
        rows.append(r)
for batch in TRANSFER:
    found = watch(batch)
    if found is None:
        continue  # still running
    path, receipts = found
    for index, receipt in sorted(receipts.items()):
        points = [(e['steps'], e['success_at_end']) for e in receipt['transfer_evals']]
        r = row(batch, receipt, str(path.relative_to(ROOT)))
        r.update(evaluation='research-side-transfer', eval_seed_base=300_000_000, final_successes=points[-1][1],
                 final_weights_sha256=None, curve=points, T80=sustained_threshold(points, 103),
                 T90=sustained_threshold(points, 116))
        if any('stages' in e for e in receipt['transfer_evals']):
            r['stages'] = [(e['steps'], e.get('stages')) for e in receipt['transfer_evals']]
        rows.append(r)
for path in {r['source'] for r in rows}:
    sources[path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
keys = [(r['batch'], r['arm'], r['seed']) for r in rows]
assert len(keys) == len(set(keys))
OUT.write_text(json.dumps(dict(runs=rows, source_sha256=dict(sorted(sources.items())),
                               note='T80/T90: first sustained crossing of 103/128 and 116/128 (next checkpoint also clears).'),
                          indent=1)+'\n')
print(json.dumps(dict(runs=len(rows), stackcube=sum(r['task'] == 'StackCube-v1' for r in rows),
                      trainer_hours=sum(r['trainer_seconds'] for r in rows)/3600)))
