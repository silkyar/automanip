"""Controlled source-policy data comparison, with no new fitting method."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.branching_validity import compare
from scripts.analyze_prefix_search import PHYSICAL, TRACE
from research.prefix_search import prefix_digest


def analyze(root, plan, dataset_output=None):
    root=Path(root); rows=[];datasets={};work=0;seconds=[];fidelity=True;enough=True
    def physical(t,prefix=False):
        return {k:v[:30 if k=='flags' else 31] if prefix else v for k,v in t.items() if k in PHYSICAL}
    for index,parent in enumerate(plan['parents']):
        banks={};entry=dict(parent=parent,banks={})
        fitting=np.random.default_rng(plan['source_split_seed']+parent).permutation(64)[:48]
        for kind in ('original','student'):
            traces={};checks=[]
            for candidate,repeat in [(c,0) for c in range(4)]+[(1,1)]:
                d=root/f'collection-{index}'/f'{kind}-candidate-{candidate}-r{repeat}'
                r=json.loads((d/'report.json').read_text())
                if r['state']!='complete' or r['transitions_completed']!=3200 or r['state_write_calls']!=0 or r['policy_noise_enabled'] is not False:
                    raise ValueError('Invalid source child')
                if (r['source_kind'],r['candidate'],r['repeat'],r['collection_index'])!=(kind,candidate,repeat,index):raise ValueError('Source identity differs')
                with np.load(d/'trace.npz',allow_pickle=False) as z:t={k:z[k] for k in TRACE}
                if prefix_digest(t)!=r['prefix_sha256']:raise ValueError('Prefix digest differs')
                traces[candidate,repeat]=t;work+=r['transitions_completed'];seconds.append(r['wall_seconds'])
            base=traces[0,0]
            for key,t in traces.items():
                c=compare(physical(base,True),physical(t,True));c.update(candidate=key[0],repeat=key[1],actions_identical=bool(np.array_equal(base['executed_actions'][:30],t['executed_actions'][:30])))
                fidelity &= c['passed'] and c['actions_identical'];checks.append(c)
            a,b=traces[1,0],traces[1,1];rep=compare(physical(a),physical(b));rep['actions_identical']=bool(np.array_equal(a['executed_actions'],b['executed_actions']))
            fidelity &= rep['passed'] and rep['actions_identical']
            y=np.stack([traces[c,0]['flags'][-1,:,6].astype(bool) for c in range(4)])
            choice=np.full(64,-1,dtype=np.int64);rng=np.random.default_rng(plan['tie_seed']+parent+(0 if kind=='original' else 100))
            for lane in range(64):
                successful=np.flatnonzero(y[:,lane])
                if len(successful):choice[lane]=0 if y[0,lane] else rng.choice(successful)
            obs=[];act=[];episodes=[]
            for lane,candidate in enumerate(choice):
                if candidate<0:continue
                t=traces[int(candidate),0];obs.extend(t['observations'][30:50,lane]);act.extend(t['executed_actions'][30:50,lane]);episodes.extend([lane]*20)
            obs=np.asarray(obs,dtype=np.float32).reshape(-1,48);act=np.asarray(act,dtype=np.float32).reshape(-1,8);episodes=np.asarray(episodes,dtype=np.int64)
            fitting_rows=int(np.isin(episodes,fitting).sum());enough &= fitting_rows>=192
            datasets[parent,kind]=dict(teacher_observations=obs,teacher_actions=act,teacher_episode=episodes,selected_candidate=choice,fitting_episode_indices=fitting)
            banks[kind]=base
            entry['banks'][kind]=dict(covered_states=int((choice>=0).sum()),candidate_successes=y.sum(1).tolist(),
                fitting_teacher_rows=fitting_rows,baseline_final_successes=int(y[0].sum()),baseline_anchor_grasp=int(base['flags'][29,:,0].sum()),baseline_anchor_placement=int(base['flags'][29,:,3].sum()),prefix_checks=checks,repeat_check=rep)
        diff=banks['student']['position'][30]-banks['original']['position'][30]
        entry['paired_anchor_position_RMS_difference_metres']=float(np.sqrt(np.square(diff).mean()))
        rows.append(entry)
    if work!=plan['max_transitions']:raise ValueError('Total source accounting differs')
    admitted=bool(fidelity and enough)
    if admitted and dataset_output is not None:
        out=Path(dataset_output);out.mkdir(exist_ok=False)
        for (parent,kind),d in datasets.items():np.savez_compressed(out/f'{kind}-{parent}.npz',**d)
    return dict(format='automanip-onpolicy-support-results-v1',state='complete',new_training_transitions=work,
        optimizer_calls=0,worker_seconds=seconds,fidelity_passed=bool(fidelity),fitting_data_sufficient=bool(enough),
        screen_passed=admitted,parents=rows,decision='Fit fixed matched aggregation students and assess' if admitted else 'Withhold fitting; no extra label queries under this registration',
        scope='One limited data-aggregation step with an imperfect sampled supervisor; not full DAgger or a learned-policy score')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset-output')
    a=p.parse_args();r=analyze(a.root,json.loads(Path(a.plan).read_text()),a.dataset_output);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('parents','worker_seconds')},indent=2))
