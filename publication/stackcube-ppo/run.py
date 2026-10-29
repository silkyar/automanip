"""Thin launcher around the unchanged ManiSkill state-PPO example."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OFFICIAL_PPO_SHA256='3fa4818861d428480244aae889bc679b2c8e89b59d6289014003c266a84dbab0'


def command(seed, official=False):
    if hashlib.sha256((ROOT/'ppo.py').read_bytes()).hexdigest()!=OFFICIAL_PPO_SHA256:
        raise ValueError('Vendored official PPO source differs from the recorded experiment')
    path=ROOT/('official.json' if official else 'recipe.json')
    recipe=json.loads(path.read_text())
    if set(recipe)!={'gamma','ent_coef','partial_reset'}:
        raise ValueError('Expected three explicit recipe choices')
    if type(recipe['partial_reset']) is not bool:
        raise ValueError('partial_reset must be a boolean')
    label='official' if official else 'recipe'
    return [sys.executable,str(ROOT/'ppo.py'),'--env-id','StackCube-v1','--seed',str(seed),
        '--num-envs','1024','--num-steps','50','--num-eval-envs','8','--num-eval-steps','50',
        '--update-epochs','8','--num-minibatches','32','--total-timesteps','25000000',
        '--control-mode','pd_joint_delta_pos','--gamma',str(recipe['gamma']),
        '--ent-coef',str(recipe['ent_coef']),
        '--partial-reset' if recipe['partial_reset'] else '--no-partial-reset',
        '--no-capture-video','--no-track','--exp-name',f'stackcube-{label}-seed-{seed}']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--official',action='store_true',help='Use the original official recipe')
    parser.add_argument('--dry-run',action='store_true',help='Print argv without importing Torch or creating a simulator')
    args=parser.parse_args();argv=command(args.seed,args.official)
    print(json.dumps({'argv':argv,'requested_steps':25000000,'complete_rollout_steps':24985600}),flush=True)
    if not args.dry_run:
        directory=ROOT/'runs'/f'stackcube-{"official" if args.official else "recipe"}-seed-{args.seed}'
        if directory.exists():
            raise SystemExit('Output exists; preserve it and use a new seed or checkout')
        subprocess.run(argv,cwd=ROOT,check=True)
