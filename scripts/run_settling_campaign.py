"""Confirm one fixed reference, then run nine preregistered development trials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from automanip.policy_io import load_policy, PolicySchema, MAX_MANIFEST_BYTES, MAX_WEIGHTS_BYTES
from automanip.research import snapshot_file, validate_campaign
from automanip.settling import campaign
from automanip.stackcube import STACK_TASK, check_frozen_protocol

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_stackcube_experiments import prepare, role_command, directory, write_json
from scripts.run_stackcube_baseline import run_child


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();out=args.output_root.resolve()
    if (out/'batch.exit').exists() or (out/'progress.json').exists():raise RuntimeError('campaign already started')
    config=validate_campaign(campaign());protocol=check_frozen_protocol()
    status=1
    progress=dict(status='starting',started_at=time.time(),confirmation_queries=0,
                  protocol_sha256=protocol,promoted=False,configuration=config)
    write_json(out/'progress.json',progress)
    try:
        users,work=prepare(out/'confirmation');evaluator=users['eval']
        plan=json.loads((out/'inputs/confirmation-plan.json').read_text())
        if [item['seed'] for item in plan]!=[1,2,3]:raise ValueError('three fixed reference checkpoints required')
        results=[]
        progress['status']='confirming_reference';write_json(out/'progress.json',progress)
        for item in plan:
            seed=item['seed'];source=out/'inputs/anchor-policies'/f'seed-{seed}'
            policy=load_policy(source,PolicySchema(STACK_TASK.fingerprint,48,8))
            if policy.weights_sha256!=item['weights_sha256']:raise ValueError('reference checkpoint digest mismatch')
            inputs=directory(work/'inputs'/f'settling-confirmation-seed-{seed}',0,evaluator.pw_gid,0o750)
            for name,limit in [('manifest.json',MAX_MANIFEST_BYTES),('weights.json',MAX_WEIGHTS_BYTES)]:
                snapshot_file(source/name,inputs/name,limit,0,evaluator.pw_gid)
            report=work/'evaluations'/f'settling-reference-seed-{seed}-confirmation.json'
            attempt=out/'confirmation'/f'seed-{seed}-attempt.json'
            # Exclusive claims prevent silently re-querying this split after a restart.
            with attempt.open('x') as stream:json.dump(dict(seed=seed,weights_sha256=policy.weights_sha256,started_at=time.time()),stream)
            command=[sys.executable,ROOT/'scripts/evaluate_stackcube.py',inputs,'--split','confirmation','--output',report]
            seconds=run_child(role_command(evaluator.pw_name,ROOT/'src',command),work/'evaluations',
                              out/'confirmation'/f'seed-{seed}.log',2400,out/'heartbeat.json')
            result=json.loads(report.read_text())
            if (result['protocol_sha256']!=protocol or result['weights_sha256']!=policy.weights_sha256 or
                result['split']!='confirmation' or result['smoke'] or result['episode_count']!=512 or result['failed_episodes']):
                raise RuntimeError('invalid confirmation report')
            snapshot_file(report,out/'confirmation'/report.name,8*1024*1024)
            results.append(dict(seed=seed,evaluation_seconds=seconds,report=result))
            progress.update(confirmation_queries=len(results)*512,confirmation_results=results)
            write_json(out/'progress.json',progress)
            print('REFERENCE_CONFIRMED',seed,result['success_at_end_rate'],flush=True)
        write_json(out/'confirmation/summary.json',dict(reference='gamma=0.9, entropy=0.003',
                   mean_success=sum(r['report']['success_at_end_rate'] for r in results)/3,
                   confirmation_queries=1536,results=results,
                   limitation='Frozen historical confirmation seeds, independent of development; not a private final holdout.'))
        # Failed execution stops the campaign; a lower scientific score is still valid evidence.
        config_path=out/'configuration.json';write_json(config_path,config)
        def train(label,smoke=False):
            command=[sys.executable,ROOT/'scripts/run_stackcube_experiments.py','--campaign',config_path,
                     '--output-root',out/label,'--execute']
            if smoke:command.append('--smoke')
            progress['status']=label;write_json(out/'progress.json',progress)
            with (out/(label+'.log')).open('w') as stream:
                subprocess.run(list(map(str,command)),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                               timeout=1800 if smoke else 24000,check=True)
        train('smoke',True)
        experiment=out/'experiments';experiment.mkdir()
        for name in ['baseline-development','anchor-development']:
            target=experiment/name;target.mkdir()
            for seed in (1,2,3):snapshot_file(out/'inputs'/name/f'seed-{seed}.json',target/f'seed-{seed}.json',8*1024*1024)
        train('experiments')
        progress.update(status='complete',finished_at=time.time(),promoted=False)
        status=0
    except Exception as exc:
        progress.update(status='failed',error=type(exc).__name__+': '+str(exc),finished_at=time.time())
        raise
    finally:
        write_json(out/'progress.json',progress)
        (out/'batch.exit').write_text(str(status)+'\n')


if __name__=='__main__':main()
