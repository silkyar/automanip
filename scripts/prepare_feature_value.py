"""Prepare a narrow clock/control study; reuse the same pinned PPO backbone."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_scratch_credit import BASE, build_sources


FIT_BLOCK='''            automanip_journal.write(json.dumps(dict(event='linear_value_fit_reserved',steps=global_step,solves=2))+'\\n')
            fitted_value, value_stats, value_coefficients = automanip_value.fit(returns,args.gamma)
            advantages = returns - fitted_value
            values.copy_(fitted_value)
            logger.add_scalar('audit/linear_value_fit_seconds',value_stats['fit_seconds'],global_step)
            logger.add_scalar('audit/crossfit_value_MSE',value_stats['crossfit_MSE'],global_step)
            logger.add_scalar('audit/neural_critic_disabled',1,global_step)
            automanip_journal.write(json.dumps(dict(event='linear_value_fit',steps=global_step,statistics=value_stats))+'\\n')
            np.savez_compressed(automanip_root/('value-'+str(global_step)+'.npz'),
                coefficients=value_coefficients.numpy(),crossfit_values=fitted_value[:,:64].detach().cpu().numpy(),
                returns=returns[:,:64].detach().cpu().numpy())

'''


def build_programs():
    base=build_sources(BASE.read_bytes())['ppo-finite-mc']
    old=json.loads((ROOT/'cumulative/experiments/scratch-credit-v1.json').read_text())['training_plan']
    if hashlib.sha256(base.encode()).hexdigest()!=old['programs']['ppo-finite-mc']:raise ValueError('Original compared MC source drift')
    template=base.replace('import torch\n','import torch\nfrom feature_value_baseline import FeatureValueBaseline\n',1)
    template=template.replace('        return self.critic(x)\n','        return torch.zeros((len(x),1),device=x.device,dtype=x.dtype)\n',1)
    needle='probs.entropy().sum(1), self.critic(x)'
    if template.count(needle)!=1:raise ValueError('Gaussian/value method changed')
    template=template.replace(needle,'probs.entropy().sum(1), torch.zeros((len(x),1),device=x.device,dtype=x.dtype)',1)
    needle='    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)\n'
    template=template.replace(needle,"    agent.critic.requires_grad_(False)  # Preserve original initialization draws; no critic MLP work.\n    automanip_value=FeatureValueBaseline(agent.actor_mean,args.num_steps,args.num_envs,'VALUE_MODE')\n"+needle,1)
    template=template.replace('        rollout_time = time.time()\n','        automanip_value.begin()\n        rollout_time = time.time()\n',1)
    template=template.replace('            # ALGO LOGIC: action logic\n','            automanip_value.active_step=step\n            # ALGO LOGIC: action logic\n',1)
    template=template.replace('                values[step] = value.flatten()\n','                values[step] = value.flatten()\n            automanip_value.active_step=None\n',1)
    needle='            advantages = returns - values\n\n'
    if template.count(needle)!=1:raise ValueError('Finite MC region changed')
    template=template.replace(needle,FIT_BLOCK,1)
    lo=template.index('                # Value loss\n');hi=template.index('                entropy_loss = entropy.mean()\n',lo)
    template=template[:lo]+"                # Value fitted in closed form on opposite episodes; no neural critic loss.\n                v_loss = torch.zeros((),device=device)\n\n"+template[hi:]
    lines=template.splitlines(keepends=True)
    for i,line in enumerate(lines):
        if line.startswith('                    automanip_cg = '):lines[i]='                    automanip_cg = 0.0  # Critic MLP disabled; linear fits counted separately.\n'
    template=''.join(lines)
    template=template.replace("        if global_step in automanip_checkpoints:\n","        automanip_event('actor_forward_rows',automanip_value.actor_forward_rows)\n        if global_step in automanip_checkpoints:\n",1)
    programs={method:template.replace('VALUE_MODE',mode) for method,mode in [('ppo-feature-value','state'),('ppo-clock-feature-value','clock')]}
    for program in programs.values():ast.parse(program)
    return programs


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    programs=build_programs()
    for name,program in programs.items():(output/(name+'.py')).write_text(program)
    (output/'feature_value_baseline.py').write_bytes((ROOT/'research/feature_value_baseline.py').read_bytes())
    raw=(ROOT/'research/discovery/amp_v1.py').read_text()
    function=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='export_actor')
    helper='import torch\nfrom torch import nn\nfrom pathlib import Path\nimport hashlib,json\nTASK_FINGERPRINT="b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f"\n\n'+ast.get_source_segment(raw,function)+'\n'
    (output/'export_helper.py').write_text(helper)
    old=json.loads((ROOT/'cumulative/experiments/scratch-credit-v1.json').read_text())['training_plan']
    if hashlib.sha256(helper.encode()).hexdigest()!=old['helper_sha256']:raise ValueError('Shared safe exporter drift')
    plan=dict(old,format='automanip-feature-value-training-v1',programs={name:hashlib.sha256(program.encode()).hexdigest() for name,program in programs.items()},
              value_helper_sha256=hashlib.sha256((output/'feature_value_baseline.py').read_bytes()).hexdigest(),
              method_change='Complete finiteMC target, PPO actor-only updates; critic is episode-crossfit257feature linear regression. Clock control uses same parameters/fits; h(t)=1-gamma^(50-t). Legacy inactive critic instantiated only to preserve paired actor/RNG initialization.',
              baseline_fit_compute='Cache features in ordinary actor forward; two CPUfloat64solves per rollout. Time and covariance operation estimates explicitly recorded. Disabled critic MLP has no forward/backward/optimizer work.')
    (output/'training-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    old_protocol=json.loads((ROOT/'cumulative/experiments/scratch-credit-v1.json').read_text())
    protocol=dict(format='automanip-feature-value-experiment-v1',created_at=time.time(),execution_authorized=True,
        user_authorization='Continue autonomous research until stopped; GPU cost explicitly not an issue',
        hypothesis='Cross-fitted current actor features with correct remaining-time structure can yield better finiteMC policy learning than identical state-only features, without a separate critic MLP.',
        evidence='Clock effect passed analytical constant-state control,12cellwhole-episode crossfit, fixed shuffled-clock control, and full146192parameter trajectory-gradient covariance diagnostic. None is itself a task improvement.',
        training_plan=plan,compared_existing_controls='cumulative/results/scratch-credit-20261030.json: exact original GAE and neuralMC procedures/seeds. Their results remain unchanged; no unneeded reruns.',
        comparison='Both NEWmethods remove the neural critic and use identical actor-only PPO/linear fitting. Their only difference is clock versus state-only value features. Gains over old neural baselines bundle critic fitting/objective changes; do not attribute those solely to clock.',
        development_seed_bases=old_protocol['development_seed_bases'],episodes_per_checkpoint=128,
        primary='Per-seed paired successAUC on common prespecified transition prefixes versus new state-only control AND existing strongGAE; report complete curves and600sactualendpoints. T80/T90require2consecutiveobservedcrossings; censor rather than invent ratios.',
        material_screen='Clock method must improve mean normalizedAUC by>=.10 against BOTH references, without worse common-endpoint mean success. Assess all3seedpairs. No10xclaim without measured crossings, fullcomputecomparison and laterheldtaskreplication.',
        compute='Same600s/10.24Mceilings, sameL40S/4CPU/16GB. Report actual actorforward/backward rows, optimizercalls, linearfit time/solves/FLOPestimates, totalGPUworker time and blindtrace/export overhead. Extra linear fitting is counted; removed neuralcritic work is explicit.',
        actor_interface='Unchanged48stateinputs/8rawjointdelta outputs,3x256tanh. Clock affects training value baseline only; no clock or learned value model enters exported actor/evaluator.',
        source_acquisition_transitions=0,max_training_transitions=61440000,max_public_scoring_transitions=307200,
        final_queries=0,confirmation_queries=0,followup='Reassess after this one mechanism study; no automatic larger budgets or parameter/architecture sweep.')
    (output.parent/'scientific-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(programs=plan['programs'],helper=plan['value_helper_sha256'],new_training_launched=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
