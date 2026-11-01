"""Run three finite, evidence-gated PPO search phases after the current sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from automanip.overnight import campaign, candidate_stats, choose_tuning, choose_efficiency
from automanip.research import canonical_digest, validate_campaign


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--staged-source',type=Path,required=True)
    args=parser.parse_args()
    parent=args.parent.resolve();out=parent/'overnight'
    out.mkdir(exist_ok=True)
    root=Path('/opt/automanip-credit-20261026/source')
    status=1
    state=dict(status='waiting_for_credit_sweep',started_at=time.time(),max_additional_runs=24,
               confirmation_queries=0,promoted=False,phases=[],stopping_rules={
                   'hard_deadline':'16 hours after original Pod start, independently supervised',
                   'max_total_pod_compute_usd':10,'max_hourly_rate_used':.6,
                   'plateau':'No meaningful success or speed improvement in the half-budget phase',
                   'error':'Stop on any failed training, artifact, protocol or evaluator check'})
    save(out/'progress.json',state)
    try:
        # Never modify the source used by the active sweep while it is running.
        while not (parent/'batch.exit').exists():time.sleep(10)
        if (parent/'batch.exit').read_text().strip()!='0':raise RuntimeError('original sweep failed')
        original=json.loads((parent/'summary.json').read_text())
        anchor=next(item for item in candidate_stats(original) if item['candidate']['id']=='gamma090')
        anchor['candidate']=campaign('short')['candidates'][0]
        state['anchor']=anchor
        manifest=json.loads((args.staged_source/'source-manifest.json').read_text())
        for name,digest in manifest.items():
            if hashlib.sha256((args.staged_source/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('staged source digest mismatch: '+name)
        # Keep the original source snapshot on the network volume unchanged.
        shutil.copytree(args.staged_source,root,dirs_exist_ok=True)
        os.environ.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',
                          OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        os.chdir(root)
        py=Path('/opt/automanip-stackcube-venv/bin/python')
        def run_phase(config,label,smoke=False):
            validate_campaign(config)
            target=out/label;target.mkdir()
            config_path=target/'configuration.json';save(config_path,config)
            save(target/'source-digest.json',dict(manifest_sha256=canonical_digest(manifest)))
            if not smoke:
                shutil.copytree(parent/'baseline-development',target/'baseline-development')
                anchor_dir=target/'anchor-development';anchor_dir.mkdir()
                for run in original['runs']:
                    if run['candidate']['id']=='gamma090':save(anchor_dir/f"seed-{run['seed']}.json",run['final']['report'])
            command=[str(py),'scripts/run_stackcube_experiments.py','--campaign',str(config_path),
                     '--output-root',str(target),'--execute']
            if smoke:command.append('--smoke')
            state.update(status='running',current_phase=label)
            state['phases'].append(dict(label=label,configuration=config,configuration_sha256=canonical_digest(config),started_at=time.time()))
            save(out/'progress.json',state)
            with (target/'controller.log').open('w') as log:
                subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               timeout=1200 if smoke else 27000,check=True)
            state['phases'][-1]['finished_at']=time.time();save(out/'progress.json',state)
            return json.loads((target/'summary.json').read_text())
        tune=campaign('tune')
        # Different smoke IDs/budget; this does not enter scientific summaries.
        run_phase(tune,'smoke',smoke=True)
        tuned=run_phase(tune,'tune')
        chosen=choose_tuning(tuned,anchor)
        base=chosen['candidate']['id'].split('-')[1]
        state.update(tuning_choice=chosen,tuning_improved=chosen is not anchor)
        save(out/'progress.json',state)
        short=run_phase(campaign('short',base),'short')
        efficient=choose_efficiency(short)
        state['efficiency_choice']=efficient
        save(out/'progress.json',state)
        if efficient:
            _,base,variant=efficient['candidate']['id'].split('-')
            verified=run_phase(campaign('verify',base,variant),'verify')
            state['verification']=candidate_stats(verified)
            state['stop_reason']='Finite search completed; full-budget follow-up measured across three seeds'
        else:
            state['stop_reason']='Plateau: neither half-budget change met success/speed criteria'
        state.update(status='complete',finished_at=time.time(),confirmation_queries=0,promoted=False)
        status=0
    except Exception as exc:
        state.update(status='failed',finished_at=time.time(),error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        save(out/'progress.json',state)
        (parent/'overnight.exit').write_text(str(status)+'\n')


if __name__=='__main__':main()
