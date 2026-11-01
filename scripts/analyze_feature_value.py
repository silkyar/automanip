"""Account for the completed feature-value study without rerunning policies."""
import json
from pathlib import Path
from scripts.analyze_scratch_credit import analyze

ROOT=Path(__file__).resolve().parents[1]
METHODS=['ppo-feature-value','ppo-clock-feature-value']


def main():
    root=ROOT/'artifacts/cumulative/feature-value-20261030'
    reports=[json.loads((root/f'scores/trial-{i}/report.json').read_text()) for i in range(6)]
    receipts=[json.loads((root/f'training/trial-{i}/receipt.json').read_text()) for i in range(6)]
    compute=json.loads((root/'learner-compute.json').read_text())['trials']
    result=analyze(reports,receipts,methods=METHODS)
    result['format']='automanip-feature-value-analysis-v1'
    old=json.loads((ROOT/'cumulative/results/scratch-credit-20261030.json').read_text())
    strong=[t for t in old['trials'] if t['method']=='ppo-gae']
    clock=[r for r in reports if r['method']==METHODS[1]]
    strong_reports=[dict(state='complete',seed=t['seed'],method='ppo-gae',points=t['learning_curve'],final_queries=0,confirmation_queries=0) for t in strong]
    strong_receipts=[dict(state='complete',seed=t['seed'],method='ppo-gae',actual_work=dict(environment_completed=t['training_transitions'],optimizer_completed=t['optimizer_calls']),trainer_process_seconds=t['trainer_process_seconds'],wrapper_seconds=t['wrapper_seconds']) for t in strong]
    result['versus_strong_gae']=analyze(strong_reports+clock,strong_receipts+[r for r in receipts if r['method']==METHODS[1]],methods=['ppo-gae',METHODS[1]])
    result['registered_material_screen_passed']=all(a['mean_paired_auc_gain']>=.10 and a['final_success_point_preserved_all_seeds'] for a in [result,result['versus_strong_gae']])
    trials=[]
    for i,(r,c) in enumerate(zip(reports,receipts)):
        assert compute[i]['index']==i and compute[i]['method']==r['method']
        assert len(r['points'])==8 and all(p['failures']==0 for p in r['points'])
        trials.append(dict(index=i,method=r['method'],seed=r['seed'],trainer_process_seconds=c['trainer_process_seconds'],wrapper_seconds=c['wrapper_seconds'],training_transitions=c['actual_work']['environment_completed'],optimizer_calls=c['actual_work']['optimizer_completed'],learner_seconds=compute[i]['learner_seconds'],PPO_seconds=compute[i]['PPO_seconds'],linear_fit_seconds=compute[i]['linear_fit_seconds'],linear_solves=c['linear_value_solves_completed_confirmed'],rollout_and_trace_seconds=compute[i]['rollout_and_trace_seconds'],learning_curve=[{k:v for k,v in p.items() if k!='outcomes'} for p in r['points']],source_acquisition_transitions=0,confirmation_queries=0,final_queries=0,actual_critic_mlp_forward_rows=c['actual_critic_mlp_forward_rows'],actor_forward_rows=c['actor_forward_rows_confirmed']))
    summary=dict(format='automanip-feature-value-results-v1',state='complete',scientific_training_runs=6,new_training_transitions=sum(t['training_transitions'] for t in trials),new_scoring_transitions=sum(r['environment_completed'] for r in reports),source_acquisition_transitions=0,trials=trials,analysis=result,decision='Rejected within the registered budget. Both methods scored0/128 at every checkpoint in every seed. Offline covariance reduction did not establish a useful online learning mechanism.',protected_benchmark_replaced=False)
    for path,data in [(root/'analysis-v1.json',result),(ROOT/'cumulative/results/feature-value-20261030.json',summary)]:
        with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
    status=ROOT/'cumulative/results/feature-value-20261030-status.json'
    status.write_text(json.dumps(dict(state='complete',scientific_training_runs_completed_in_this_study=6,result='cumulative/results/feature-value-20261030.json',training_transitions=summary['new_training_transitions'],scoring_transitions=summary['new_scoring_transitions']),indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('trials','analysis')},indent=2))


if __name__=='__main__':main()
