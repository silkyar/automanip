"""Emit a reviewable six-job Modal plan; this command never launches compute."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REVIEWED={
 'full-horizon-control':'68e09965db1378616f380eb763862149d8c28232075e06741426caba3dfb5064',
 'full-horizon-capg':'8a95fca4ac08a8118ecb361fb9a458e137578673a1b4fcfe2c94fdbd3033b971',
}


def make_plan(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['trainer_sha256']!=REVIEWED:raise ValueError('unreviewed payload registry')
    experiment=manifest['experiment']
    if experiment['max_training_seconds']!=1320 or experiment['training_seeds']!=[1,2,3] or experiment['confirmation_queries']!=0:
        raise ValueError('experiment budget/seeds changed')
    jobs=[]
    for seed in [1,2,3]:
        for method,digest in REVIEWED.items():
            source=directory/(method+'.py')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('payload digest changed')
            jobs.append({'job_id':f'capg-{method}-seed-{seed}','method':method,'training_seed':seed,
                         'trainer_sha256':digest,'gpu':'L4','trainer_seconds_cap':1320,
                         'requested_transition_ceiling':25_000_000,'evaluation_split':'development',
                         'confirmation_queries':0,'absolute_deadline_policy':'trusted coordinator fixes one epoch lease before invocation; every attempt inherits it',
                         'interruption_policy':'record interruptions; no automatic new lease or silent retry',
                         'training_started':False})
    return {'experiment':experiment,'max_parallel_trainers':3,'jobs':jobs,
            'launch_enabled':False,'reason':'full trial launcher requires review after the trusted smoke'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payloads',type=Path,default=ROOT/'.runtime/modal-capg-payloads')
    p.add_argument('--output',type=Path,default=ROOT/'.runtime/modal-capg-six-job-plan.json')
    args=p.parse_args();args.output.write_text(json.dumps(make_plan(args.payloads),indent=2)+'\n')
    print(args.output)
