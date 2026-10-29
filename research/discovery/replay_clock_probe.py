"""Exact scalar soft-Bellman/Polyak clock check, not a neural reconstruction.

Assume the online scalar critic fits its current bootstrap target instantly.
No entropy/policy/data drift: this isolates the target-backup relaxation clock.
"""
import hashlib,json,math,time
from pathlib import Path
import numpy as np

start=time.monotonic();reward=.05;rows=[]
for name,gamma,tau,updates_per_transition,delay,lr in [
    ('historical-recipe-clock',.9,.01,.5,1,.0003),
    ('fast-recipe-clock',.99,.005,1/16,3,.001),
    ('fast-with-only-discount-restored',.9,.005,1/16,3,.001)]:
    k=1-tau*(1-gamma);qstar=reward/(1-gamma)
    half_updates=math.log(.5)/math.log(k);half_transitions=half_updates/updates_per_transition
    target=0.;checkpoints={};iterations=64000 if updates_per_transition==1/16 else 512000
    for update in range(1,iterations+1):
        online=reward+gamma*target
        target=(1-tau)*target+tau*online
        transitions=update/updates_per_transition
        if transitions in (51200,102400,256000,512000,1024000):
            exact=qstar*(1-k**update)
            assert abs(target-exact)<1e-9
            checkpoints[str(int(transitions))]=dict(target=target,true_fixed_point=qstar,
                target_fraction_of_fixed_point=target/qstar)
    rows.append(dict(name=name,gamma=gamma,tau=tau,critic_updates_per_transition=updates_per_transition,
        policy_delay=delay,learning_rate=lr,relaxation_eigenvalue_per_critic_update=k,
        target_half_life_critic_updates=half_updates,target_half_life_transitions=half_transitions,
        actor_or_temperature_lr_times_updates_per_transition=lr*updates_per_transition/delay,
        checkpoints=checkpoints))
old,new,restored=rows
result=dict(state='passed',seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scalar_assignments=sum(64000 if x['critic_updates_per_transition']==1/16 else 512000 for x in rows),
    neural_optimizer_calls=0,robot_transitions=0,records=rows,
    fast_vs_historical_backup_half_life_ratio=new['target_half_life_transitions']/old['target_half_life_transitions'],
    discount_restored_vs_historical_half_life_ratio=restored['target_half_life_transitions']/old['target_half_life_transitions'],
    historical_vs_fast_actor_lr_clock_ratio=old['actor_or_temperature_lr_times_updates_per_transition']/new['actor_or_temperature_lr_times_updates_per_transition'],
    limits='Instantaneously fitted scalar critic, fixed reward/policy, no entropy, features, Adam, replay or actor improvement. Clock plausibility only; cannot identify the robot regression cause.',
    conclusion='Changing update cadence, discount and target EMA together changes bootstrap relaxation in interaction units by about160x; restoring discount alone leaves about16x difference. No robot causal or efficiency claim.')
p=Path(__file__).with_name('replay-clock-probe-20261028.json')
if p.exists():raise FileExistsError('Preserve receipt')
p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
