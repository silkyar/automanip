"""Build the human experiment index from recorded aggregates, never run policies."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT=Path(__file__).resolve().parents[1]
rows=[]
sources={}


def read(name):
    raw=(ROOT/name).read_bytes()
    sources[name]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def add(family,seed,seconds,steps,report,source,*,task='StackCube-v1',gpu='L4',prefix=None):
    rows.append(dict(experiment=family,task=task,seed=seed,trainer_seconds=seconds,
                     environment_transitions=steps,gpu=gpu,initialization='continuation' if prefix else 'random',
                     prefix=prefix,development_successes=report['successes_at_end'],development_episodes=128,
                     confirmation_successes=None,confirmation_episodes=None,weights_sha256=report.get('weights_sha256'),source=source))


def confirm(family,seed,successes,episodes):
    selected=[r for r in rows if r['experiment']==family and r['seed']==seed]
    assert len(selected)==1
    selected[0].update(confirmation_successes=successes,confirmation_episodes=episodes)


name='runs/runpod-20261025/summary.json'
for r in read(name)['runs']:
    add('PickCube adapted PPO',r['training_seed'],r['training_seconds'],r['environment_transitions'],r,name,task='PickCube-v1')
name='runs/stackcube-20261026/summary.json'
for r in read(name)['runs']:
    add('StackCube official PPO',r['seed'],r['trainer_process_seconds'],r['environment_transitions'],r['final_evaluation']['development'],name)
    report=r['final_evaluation']['confirmation'];confirm('StackCube official PPO',r['seed'],report['successes_at_end'],512)
for name in ('runs/stackcube-research-20261027/summary.json','runs/stackcube-research-20261027/overnight/tune/summary.json','runs/stackcube-research-20261027/overnight/short/summary.json'):
    for r in read(name)['runs']:
        add(r['candidate']['id'],r['seed'],r['trainer_process_seconds'],r['environment_transitions'],r['final']['report'],name)
archive='.runtime/settling-20261027-results.tar.gz'
member='automanip-settling-20261027/experiments/summary.json'
with tarfile.open(ROOT/archive,'r:gz') as bundle:
    entry=bundle.getmember(member)
    assert entry.isfile() and entry.size<2*1024*1024
    raw=bundle.extractfile(entry).read(entry.size+1)
settling=json.loads(raw);source=archive+'#'+member;sources[source]=hashlib.sha256(raw).hexdigest()
for r in settling['runs']:
    add(r['candidate']['id'],r['seed'],r['trainer_process_seconds'],r['environment_transitions'],r['final']['report'],source)
for r in read('runs/stackcube-settling-20261027/confirmation/summary.json')['results']:
    confirm('tune-entropy003-control',r['seed'],r['report']['successes_at_end'],512)
for r in read('runs/private-confirmation/stackcube-full-horizon-20261027/verified-summary.json')['per_seed']:
    confirm('settle-full-horizon',r['seed'],r['successes'],512)
for name in ('runs/stackcube-capg-screen-20261027/summary.json','runs/stackcube-precision-screen-20261027/summary.json','runs/stackcube-precision-replication-20261027/summary.json','runs/stackcube-risk-continuation-20261028/summary.json'):
    summary=read(name)
    for r in summary['training']:
        job=r['job'];method=job['method'];key=job['job_id'] if 'replication' in name else method
        point=summary['development_curves'][key][-1]
        add(method,job['seed'],r['trainer_process_seconds'],r['actual_steps'],
            dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),name,
            gpu='L40S',prefix=summary.get('training_prefix'))
name='runs/stackcube-sac-screen-seed1-20261027/summary.json';summary=read(name);r=summary['training'];point=summary['development_curve'][-1]
add('SAC adapted',1,r['trainer_process_seconds'],r['actual_steps'],dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),name,gpu='L40S')
confirmation=ROOT/'runs/private-confirmation/stackcube-precision-20261028/summary.json'
if confirmation.exists():
    for r in read(str(confirmation.relative_to(ROOT)))['records']:
        confirm('precision-constant',r['seed'],r['report']['successes_at_end'],512)
assert len(rows)==56 and len({(r['experiment'],r['seed']) for r in rows})==56

bc_archive='runs/stackcube-bc-exploration-pilot-20261028/summary.json'
if (ROOT/bc_archive).exists():
    bc_summary=read(bc_archive)
    if bc_summary['state']=='complete':
        prefix=dict(bc_summary['teacher_prefix'])
        prefix['demo_transitions']=bc_summary['stages']['collect']['collection']['environment_transitions']
        prefix['bc_trainer_seconds']=bc_summary['stages']['bc']['trainer_process_seconds']
        for stage,family in [('bc','BC-only random student'),('assist-bc-ppo','BC plus standard PPO'),('assist-bc-ppo-residual-std','BC plus residual exploration PPO')]:
            native=bc_summary['stages'][stage]
            evaluation=bc_summary['evaluations']['bc-only' if stage=='bc' else stage]
            add(family,1,native['trainer_process_seconds'],native.get('actual_steps',0),dict(successes_at_end=evaluation['successes'],weights_sha256=evaluation['weights_sha256']),bc_archive,gpu='L40S',prefix=prefix)
            rows[-1].update(demonstration_assisted=True,initialization='random student behavior cloning' if stage=='bc' else 'shared BC actor; fresh critic and optimizer',bc_updates=bc_summary['stages']['bc']['bc']['optimizer_updates'])

kl_archive='runs/stackcube-kl-value-screen-20261028/reconciled-summary.json'
if (ROOT/kl_archive).exists():
    kl_summary=read(kl_archive)
    if kl_summary['state']=='complete':
        for native in kl_summary['training']:
            item=native['job'];point=kl_summary['development_curves'][item['method']][-1]
            add(item['method'],item['seed'],native['trainer_process_seconds'],native['actual_steps'],dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),kl_archive,gpu='L40S')
            rows[-1].update(optimizer_calls=native['optimizer_step_count'],joint_optimizer_calls=native['measured_joint_optimizer_calls'],critic_only_optimizer_calls=native['measured_critic_only_optimizer_calls'])

replay_archive='runs/stackcube-replay-moments-screen-20261028/trusted-summary.json'
if (ROOT/replay_archive).exists():
    replay_summary=read(replay_archive)
    if replay_summary['state']=='complete':
        for native in replay_summary['training']:
            method=native['method'];report=native['report']
            curve=replay_summary['development_curves'][method];point=curve[-1]
            assert point['steps']==report['counts']['transitions']
            add('replay-'+method,report['config']['seed'],native['trainer_process_seconds'],
                report['counts']['transitions'],dict(successes_at_end=point['successes'],
                weights_sha256=point['weights_sha256']),replay_archive,gpu='L40S')
            rows[-1].update(learning_curve=curve,optimizer_work=report['optimizer_work'],
                stopped_by=report['stopped_by'],requested_transitions=report['config']['total_transitions'],
                comparison_scope='Single-seed matched-update moment mechanism screen; not official CrossQ reproduction or tuned SAC claim')

gamma_archive='runs/gamma-sufficiency-20261028/trusted-summary.json'
if (ROOT/gamma_archive).exists():
    summary=read(gamma_archive)
    if summary['state']=='complete':
        native=summary['training'];report=native['report'];point=summary['development_curve'][-1]
        add('replay-gamma090-sufficiency',1,native['trainer_process_seconds'],report['counts']['transitions'],
            dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),gamma_archive,gpu='L40S')
        rows[-1].update(learning_curve=summary['development_curve'],primary_verdict=summary['primary_verdict'],
            optimizer_work=report['optimizer_work'],comparison_scope='Single-factor gamma restoration; early grasp/lift gate, not stack efficiency')

temperature_archive='runs/temperature-sufficiency-20261028/trusted-summary.json'
if (ROOT/temperature_archive).exists():
    summary=read(temperature_archive)
    if summary['state']=='complete':
        native=summary['training'];report=native['report'];point=summary['development_curve'][-1]
        add('replay-initial-alpha005-sufficiency',1,native['trainer_process_seconds'],report['counts']['transitions'],
            dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),temperature_archive,gpu='L40S')
        rows[-1].update(learning_curve=summary['development_curve'],primary_verdict=summary['primary_verdict'],
            optimizer_work=report['optimizer_work'],initial_alpha=.005,
            comparison_scope='Single-factor initialalpha restoration; early grasp/lift gate partial; no stack efficiency claim')

route_archive='runs/entropy-route-20261029/reconciled-summary.json'
if not (ROOT/route_archive).exists():route_archive='runs/entropy-route-20261029/trusted-summary.json'
route_attempts=[]
if (ROOT/route_archive).exists():
    summary=read(route_archive)
    for route,arm in summary['arms'].items():
        native=arm.get('training',{})
        report=native.get('report',{})
        point=arm.get('development')
        route_attempts.append(dict(route=route,state=arm['state'],source=route_archive,
            actual_transitions=report.get('counts',{}).get('transitions'),
            trainer_seconds=native.get('trainer_process_seconds'),error=arm.get('error')))
        if arm['state']!='complete' or point is None:continue
        assert point['steps']==report['counts']['transitions']==102400
        add('entropy-route-'+route,1,native['trainer_process_seconds'],report['counts']['transitions'],
            dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),route_archive,gpu='L40S')
        rows[-1].update(learning_curve=[point],primary_verdict=arm['primary_verdict'],
            optimizer_work=report['optimizer_work'],
            comparison_scope='Adaptive actor/soft-target entropy-route sufficiency; one seed and grasp endpoint, not fixed-alpha factorial or stack-efficiency claim')

behavior_archive='runs/joint-exploration-20261029/reconciled-summary.json'
if (ROOT/behavior_archive).exists():
    summary=read(behavior_archive)
    for mode,arm in summary['arms'].items():
        if arm['state']!='complete' or 'development' not in arm:continue
        native=arm['training'];report=native['report'];point=arm['development']
        assert point['steps']==report['counts']['transitions']==102400
        add('behavior-'+mode+'-uniform',1,native['trainer_process_seconds'],102400,
            dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),behavior_archive,gpu='L40S')
        rows[-1].update(learning_curve=[point],study_primary_verdict=summary['primary_verdict'],
            optimizer_work=report['optimizer_work'],collector_work=report['collector_work'],
            comparison_scope='Behavior-only matched-coordinate-marginal study; joint hypothesis failed; coordinate early lift signal unconfirmed, no stacking gain')

curve_archive='runs/full-stacking-curve-20261029/reconciled-summary.json'
if (ROOT/curve_archive).exists():
    summary=read(curve_archive)
    for mode,arm in summary['arms'].items():
        if arm['state']!='complete':continue
        native=arm['training'];report=native['report'];point=arm['development']
        add('full-task-'+mode+'-SAC',2,native['trainer_process_seconds'],report['counts']['transitions'],
            dict(successes_at_end=point['successes'],weights_sha256=point['weights_sha256']),curve_archive,gpu='L40S')
        rows[-1].update(learning_curve=arm['learning_curve'],study_primary_verdict=summary['primary_verdict'],
            optimizer_work=report['optimizer_work'],collector_work=report['collector_work'],
            comparison_scope='Fresh seed2 full-task curves, matched512k primary and dual600s/1.024M ceilings; single-seed screen, historical PPO context only')

descriptions={
 'PickCube adapted PPO':'State PPO, 600-second budget; stock pick metric',
 'StackCube official PPO':'Unmodified official PPO; discount .8, entropy 0',
 'gamma090':'Discount .8 → .9', 'gamma095':'Discount .8 → .95',
 'tune-gamma0875-control':'Discount .875; entropy 0', 'tune-gamma0925-control':'Discount .925; entropy 0',
 'tune-entropy001-control':'Discount .9; entropy .001', 'tune-entropy003-control':'Discount .9; entropy .003',
 'short-entropy003-control':'Selected recipe, half the transition budget',
 'short-entropy003-lr2':'Half budget; double learning rate to .0006',
 'short-entropy003-epochs4':'Half budget; 4 PPO epochs instead of 8',
 'settle-timed-control':'Fresh .9/.003 control under 22-minute ceiling',
 'settle-full-horizon':'Keep training episodes to horizon after success',
 'settle-entropy-decay':'Entropy .003 → 0 after 15M; original success resets',
 'settle-gamma0925-entropy003':'Discount .925 plus entropy .003',
 'full-horizon-control':'Matched L40S full-horizon PPO control',
 'full-horizon-capg':'Boundary-probability likelihood for clipped actions',
 'SAC adapted':'Replay-based SAC with safe-schema-compatible actor',
 'precision-constant':'Full-horizon PPO, twice the transition budget',
 'precision-entropy-anneal':'50M; reduce entropy to zero after 25.6M',
 'precision-lr-anneal':'50M; reduce LR to .00003 after 25.6M',
 'risk-ppo-control':'Continue own 25M policy using ordinary PPO',
 'risk-episode-mean':'Same continuation; episode-mean score gradient',
 'risk-episode-cvar10':'Same continuation; lower-tail CVaR10 score gradient',
 'BC-only random student':'Imitate own teacher demonstrations; no student RL',
 'BC plus standard PPO':'Shared clone; ordinary PPO exploration',
 'BC plus residual exploration PPO':'Same clone; initial noise from training-demo errors',
 'scratch-kl-stop-control':'Fresh strong PPO control; original joint KL stop',
 'scratch-kl-stop-critic':'Continue critic only after actor KL stop',
 'replay-target-sac':'Adapted target-network replay comparator; matched actor and update cadence',
 'replay-joint-moments':'Target-free critic; joint current/next normalization moments',
 'replay-split-moments':'Same target-free critic; split current/next moments, pooled EMA clock',
 'replay-gamma090-sufficiency':'Fast target-SAC gamma .99 → .9 only; smaller early-stage diagnostic budget',
 'replay-initial-alpha005-sufficiency':'Fast gamma .99 target-SAC initial alpha1 → .005 only',
 'entropy-route-high-actor-low-backup':'Original high actor/temperature alpha; scale only soft backup entropy by .005',
 'entropy-route-low-actor-high-backup':'Original low actor/temperature alpha; scale only soft backup entropy by200',
 'behavior-joint-uniform':'Original low-alpha SAC learner;10% joint-vector uniform behavior mixture',
 'behavior-coordinate-uniform':'Same learner and conditional coordinate marginals;10% independent-coordinate uniform mixture',
 'full-task-baseline-SAC':'Fresh seed2 ordinary low-alpha SAC; full-task curve,600s/1.024M dual ceilings',
 'full-task-coordinate-SAC':'Same fresh seed2 learner;10% coordinate mixture; full-task curve, same dual ceilings',
}
groups=collections.OrderedDict()
for r in rows:groups.setdefault(r['experiment'],[]).append(r)
assert set(groups)<=set(descriptions)
for r in rows:r['change']=descriptions[r['experiment']]
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
mechanism_probes=[]
probe_source='runs/stackcube-first-update-probe-20261028/run/report.json'
if (ROOT/probe_source).exists():
    probe=read(probe_source)
    if probe['state']=='complete':
        mechanism_probes.append(dict(experiment=probe['experiment'],task='StackCube-v1',
            probe_process_seconds=probe['elapsed_seconds'],environment_transitions=probe['progress']['new_simulator_transitions'],
            initialization='same assisted actor; three collection draws, not independent training seeds',
            optimizer_calls=probe['optimizer_calls'],forward_work=probe['forward_work'],
            source=probe_source,benchmark_comparison=False,development_queries=0,confirmation_queries=0))
probe_source='runs/lift-control-20261028/run/report.json'
if (ROOT/probe_source).exists():
    probe=read(probe_source)
    mechanism_probes.append(dict(experiment='lift-control-20261028',state=probe['state'],
        task='StackCube-v1',probe_process_seconds=probe['elapsed_seconds'],
        environment_transitions=probe['actual_public_transitions'],attempted_transitions=probe['attempted_public_transitions'],
        optimizer_calls=probe['optimizer_calls'],paired_validity=probe['validity_passed'],
        source=probe_source,benchmark_comparison=False,development_queries=0,confirmation_queries=0))
probe_source='research/discovery/actor-gradient-snapshot-20261029.json'
if (ROOT/probe_source).exists():
    probe=read(probe_source)
    mechanism_probes.append(dict(experiment='actor-gradient-snapshot-20261029',state=probe['state'],
        task='StackCube-v1',probe_process_seconds=probe['elapsed_seconds'],environment_transitions=0,
        optimizer_calls=0,actor_forward_rows=probe['actor_forward_rows'],
        safe_numpy_policy_forward_rows=probe['safe_numpy_policy_forward_rows'],
        twin_critic_forward_rows=probe['twin_critic_forward_rows'],autograd_requests_rows=probe['autograd_requests_rows'],
        source=probe_source,benchmark_comparison=False,development_queries=0,confirmation_queries=0))
probe_source='research/discovery/coordinate-entropy-gate-20261029.json'
if (ROOT/probe_source).exists():
    probe=read(probe_source)
    mechanism_probes.append(dict(experiment='coordinate-entropy-gate-20261029',state=probe['state'],
        task='StackCube-v1',probe_process_seconds=probe['elapsed_seconds'],environment_transitions=0,
        optimizer_calls=0,actor_forward_rows=probe['actor_forward_rows'],
        quadrature_integrand_evaluations=probe['quadrature_log_Jacobian_evaluations'],
        blind_spot_gate_passed=any(row['blind_spot_gate'] for row in probe['coordinates']),
        source=probe_source,benchmark_comparison=False,development_queries=0,confirmation_queries=0))
probe_source='runs/randomized-lift-20261029/run/report.json'
if (ROOT/probe_source).exists():
    probe=read(probe_source)
    mechanism_probes.append(dict(experiment='randomized-lift-20261029',state=probe['state'],task='StackCube-v1',
        probe_process_seconds=probe['elapsed_seconds'],environment_transitions=probe['actual_public_transitions'],
        attempted_transitions=probe['attempted_public_transitions'],optimizer_calls=0,
        actor_forward_rows=probe['policy_inference_rows'],analytic_FK_pose_evaluations=probe.get('analytic_FK_pose_evaluations'),
        analytic_jacobian_pose_evaluations=probe.get('analytic_jacobian_pose_evaluations'),
        source=probe_source,benchmark_comparison=False,development_queries=0,confirmation_queries=0))
machine=dict(updated_utc=now,completed_scientific_training_runs=len(rows),experiment_families=len(groups),
             scope='Recorded scientific GPU training; smoke/setup/CPU probes separate; confirmation is evaluation, not another training run.',
             total_new_training_seconds=sum(r['trainer_seconds'] for r in rows),runs=rows,
             mechanism_probes=mechanism_probes,entropy_route_attempts=route_attempts,source_sha256=sources)
(ROOT/'docs/experiment-runs.json').write_text(json.dumps(machine,indent=2)+'\n')

def pct(value):return f'{100*value:.2f}%'
def metric(group,split):
    values=[r[f'{split}_successes']/r[f'{split}_episodes'] for r in group if r[f'{split}_successes'] is not None]
    return pct(statistics.mean(values)) if values else '—'
def steps(group):
    values=[r['environment_transitions'] for r in group]
    value=f'{values[0]/1e6:.3f}M' if len(set(values))==1 else f'{min(values)/1e6:.3f}–{max(values)/1e6:.3f}M'
    if group[0].get('demonstration_assisted'):
        return ('0 RL; 25.6k demos' if values[0]==0 else value+' RL')+'; assisted*'
    return value+(' new + 24.986M prefix' if group[0]['prefix'] else '')

bc_status='Adapter preparation and root review; no outcome yet'
bc_observation=ROOT/'.runtime/modal-bc-exploration-current-summary.json'
if bc_observation.exists():
    observed=json.loads(bc_observation.read_bytes())
    stages=', '.join(name+': '+value['state'] for name,value in observed['stages'].items())
    bc_status='Cloud pipeline '+observed['state']+'; '+stages
    if observed['state']=='complete':bc_status='Complete: BC49/128, both PPO variants0/128; failed gate, stopped'
elif (ROOT/'.runtime/modal-bc-exploration-pilot-launch.json').exists():
    bc_status='Cloud pipeline submitted; awaiting stage receipt'
kl_status='Source and CPU checks complete; paired adapter under review'
if (ROOT/'.runtime/modal-kl-value-screen-launch.json').exists():
    kl_status='Cloud pair submitted; frozen development results pending'
if (ROOT/'.runtime/modal-kl-value-terminal-summary.json').exists():
    kl_status='Complete: control98/128 versus critic29/128 at10.24M; negative gate, stopped'
lines=['# AutoManip experiment ledger','',f'Updated {now[:10]}. **{len(rows)} completed scientific training runs across {len(groups)} experiment families.**'+f" This comprises {sum(r['environment_transitions']>0 for r in rows)} RL runs and {sum(r['environment_transitions']==0 for r in rows)} behavior-cloning fits; collection, scoring and smokes are separate.",
       'The current fixed 50M StackCube recipe scores **98.70% development and 95.90% independent confirmation** across three training seeds. The research objective is now at least **10× fewer interactions to 80–90% success** against strong matched baselines, with total compute and assistance accounted for. The user authorized continuous investigation until asked to stop and explicitly resumed bounded Modal work on Oct29, with cheap mechanism gates before bounded robot work. The completed [first-update mechanism probe](first-update-probe-results-20261028.md), [scientific diagnosis](research-strategy-diagnosis-20261028.md) and [37-paper literature map](literature/survey-20261028.md) remain the research context.',
       '', 'Training time below is the measured trainer process per seed, including startup and built-in diagnostics. Independent scoring, capacity queues, image setup and orchestration are separate. A mean across three seeds is not a claim that one run took three times that long. All tasks use state observations; no vision training has run.',
       '', '## Completed training experiments','',
       'Development uses 128 fixed episodes per policy; confirmation uses 512 independent episodes. A dash means no confirmation was queried. Each name links to its recorded results. Rows with three seeds show mean trainer time and mean success; exact seed records follow below.', '',
       '| Task | Experiment and change | Seeds | GPU | Mean minutes per seed | New steps per seed | Development | Confirmation |',
       '|---|---|---:|---|---:|---|---:|---:|']
for family,group in groups.items():
    source=group[0]['source'].split('#')[0]
    label=f'[{family}](../{source}) — {descriptions[family]}'
    lines.append(f"| {group[0]['task'].replace('-v1','')} | {label} | {len(group)} | {group[0]['gpu']} | {statistics.mean(r['trainer_seconds'] for r in group)/60:.2f} | {steps(group)} | {metric(group,'development')} | {metric(group,'confirmation')} |")
lines += ['', '## What changed performance','',
 'The main measured recipe gains followed changes to discounting, entropy, training through the full horizon after success, and finally more training. These interventions do not by themselves identify the causal learning mechanism. The full-horizon 25M recipe confirmed at **94.08%** in roughly **17.8 minutes per seed** on L4. Doubling transitions to 50M confirmed at **95.90%**, with roughly **36.8 minutes per seed** on L40S. Hardware and budgets differ; this is not a controlled wall-time speedup comparison.',
 '', 'The 99.22% headline was one development seed. The three-seed development mean is 98.70%; confirmation is 95.90%. The 50M recipe missed its preregistered 99% confirmation target. Existing confirmation sets are independent of development but are historical fixed sets, not a private final holdout.',
 '', 'The half-budget recipes lost substantial performance. CAPG failed its learning-curve gate despite a mixed late-checkpoint signal. The tested SAC recipe learned grasp/transport but scored zero final stacks within its cap; this does not refute SAC generally. Entropy and LR refinement did not beat the simpler 50M constant recipe. Episode-risk continuation lost to ordinary PPO and was stopped.',
 '', 'The three risk rows are **continuations**, not fresh short runs: each starts from our own 24,985,600-step, 1,088.77-second teacher and restarts its optimizer. Times and transitions in those rows show only additional work; their shared prefix is disclosed and is already counted once in the project training history.',
 '', 'PickCube uses the stock pick metric, which does not require release, and a different controller from StackCube. Its 10-minute result is not an earliest-time-to-success measurement. StackCube uses the official frozen final-step stacking metric; ten-step stability is a separate diagnostic.',
 '', 'Assisted* rows share the **49,971,200-transition / 2,295.41-second teacher**, plus **25,600 demo transitions**, **29.37 seconds of generation** and **5,000 BC updates / 10.87 trainer seconds**. Both PPO variants then used 5.12M new transitions and fell from BC-only49/128 to0/128. Residual-noise initialization failed its gate and is stopped. Shared assistance is disclosed rather than counted as free or as a from-scratch gain.',
 '', '## Current research review and prepared work','',
 'The separately approved [first-update mechanism probe](first-update-probe-results-20261028.md) is complete and stopped: 201,600 new transitions, 46.66 seconds on one L40S, six explicitly counted one-step optimizer calls. Large mean steps damaged the assisted policy; small-step improvement direction remains unproven, so the full hypothesis gate did not pass. This is one controlled probe, not additional independently trained policies or a benchmark comparison. The later user-authorized [replay-moment study](replay-moment-study-plan-20261028.md) completed three fresh runs:1,881,024 total transitions,14 once-only development scores, all0/128, no confirmation query or replication gate passed. Different throughput censored final step counts, so shared51.2k/102.4k/256k checkpoints are the equal-step comparison. All three apps for its smoke, screen and separate public video were verified stopped with zero tasks. The two earlier critic/control records were reconciled without rerunning training. Prepared payloads below remain disabled.', '',
 '| Work | Change and budget | Status |', '|---|---|---|',
 '| [BC exploration pilot](stackcube-bc-exploration-plan-20261028.md) | Fresh GPU teacher demonstrations, one random-student BC fit, then matched standard BC+PPO versus residual-calibrated initial action noise; at most 5.12M new RL steps / 600 seconds each | '+bc_status+' |',
 '| AMP motion prior | Add adversarial transition-style reward to BC+PPO | Deferred after local temporal/OOD diagnostics; no GPU AMP training run |',
 '| [From-scratch critic completion](stackcube-kl-value-plan-20261028.md) | Stop policy at KL limit, continue value fitting on the same data; 10.24M-step / 600s ceilings | '+kl_status+' |',
 '| From-scratch update cadence | Use 512 or 256 environments instead of 1024, giving more policy updates per collected transition | Prepared but disabled under research strategy change; not launched |',
 '| Assisted mean learning rate | Scale policy-mean LR with the calibrated noise scale; critic/noise LR unchanged | Frozen proposal only; research strategy review required before any launch |',
 '', 'The BC experiment is an assisted track: it reuses our own 49,971,200-step / 2,295.41-second teacher. Teacher training, fresh demo generation, BC updates and student RL must be reported separately. It cannot establish a from-scratch sample-efficiency gain by hiding the teacher prefix.',
 '', '## Mechanism probes and execution checks','',
 'These are excluded from the scientific GPU training count above. Toy success and action MSE are not StackCube benchmark success.',
 '', '| Probe | Measured duration | Outcome |', '|---|---:|---|',
 '| [Assisted lift-control probe](lift-control-results-20261029.md) |11.47 s GPU child /15.75 s wrapper |512 public transitions; paired physical-state gate failed before any intervention. No grip/command result, fitting or protected score. Archived and stopped. |',
 '| [Frozen actor/critic gradient snapshot](actor-gradient-snapshot-results-20261029.md) |.669 s CPU |1,904 existing grasp states,16 fixed draws each. Q favors further positive saturated-joint action; entropy opposes it. Zero new interactions/fitting; no critic-truth or efficiency claim. |',
 '| [Coordinate entropy gate](coordinate-entropy-gate-results-20261029.md) |.061 s CPU |All eight hidden-coordinate gates fail; scalar conditional pressure mostly raises temperature. Analytic equal-entropy remedy costs5.05x in the declared quadratic negative control. No fitting/robot queries/method launch. |',
 '| [Randomized assisted lift probe](randomized-lift-results-20261029.md) |13.17 s GPU diagnostic /17.37 s wrapper |6,400 public transitions. Servo+closure lifts20/21eligible versus4/19servo+policygripper; allassigned1/32,1/32,4/32,20/32. No training/protectedscore or learned-efficiency claim. Archived and stopped. |',
 '| [Assisted first-update mechanism probe](first-update-probe-results-20261028.md) | 46.66 s GPU process | 201,600 total new transitions; six one-step optimizer calls /9,600 fitting rows. No-update75/192, full step38/192, bounded72/192; useful forward direction not established. No benchmark query or automatic follow-up. |',
 '| [Replay moment operator gate](research-window-replay-review-20261028.md) | .0792 s CPU | Shifted-half aliasing and gradient coupling; finite-epsilon BRN caveat. Zero robot transitions or optimizer calls. |',
 '| [Actual-critic scalar control diagnostic](research-window-replay-value-20261028.md) |17.3927 s CPU including measurement-only repeat |18,000 total optimizer calls on16,384 synthetic training rows; split normalization gives gross Q error despite small self-TD residual. Joint does not beat competent target control; not a robot result. |',
 '| Replay execution smoke |40.28 s GPU wrapper |15,360 total transitions across three tiny execution trials; six safe artifacts validated, paired initialization, finite counters. No benchmark score; excluded from scientific run count. |',
 '| Recursive credit / replay prototypes | See [discovery journal](research-discovery-20261027.md) and [neural gate](research-discovery-neural-gate-20261027.md) | CPU falsification; no robot deployment |',
 '| Learned contraction | 10.50 s | Failed smooth-system control despite hybrid-toy gain; stopped |',
 '| State-conditioned variance head | 58.91 s | Toy 88.67% versus global-variance 100%; stopped |',
 '| Quantile lower-tail objective | 10.35 s | Toy gate passed; lower reward tradeoff; motivated the later failed robot risk pilot |',
 '| Synthetic BC / discriminator component checks | 1.64 s | Dataflow gate passed across six seeds; no robot-performance claim |',
 '| Local CPU teacher collection | 14.01 s | 256 complete episodes, 12,800 transitions, 247 successful / 9 failed, all retained |',
 '| Local CPU BC | 11.77 s | 5,000 updates, held-out demo action MSE .01017 |',
 '| Fresh local CPU rollout probe | 8.57 s across three policies | Random 0/64, BC 21/64, teacher 61/64; Mac SAPIEN differs from frozen GPU stack |',
 '| Real-transition discriminator | .90 s | Weak temporal sensitivity; high style on unsuccessful OOD random policies; AMP deferred |',
 '| Residual-noise calibration | .34 s | Held-out expected clipped-action MSE .26744 → .01613; no RL gain established; low noise can suppress corrections |',
 '', 'Simulator smokes, policy-converter parity, provider compatibility and isolation checks are execution validation, not additional successful research experiments. Preserved failures include a SAC replay dtype error and an SDK image-option error before confirmation submission. No failed run is converted into a benchmark success. Current task/evaluator/seeds/physics/interfaces are unchanged.',
 '', '## Every completed training run','',
 '| Experiment | Seed | Trainer seconds | New transitions | Development | Confirmation |', '|---|---:|---:|---:|---:|---:|']
for family,group in groups.items():
    for r in sorted(group,key=lambda r:r['seed']):
        conf='—' if r['confirmation_successes'] is None else f"{r['confirmation_successes']}/{r['confirmation_episodes']}"
        lines.append(f"| {family} | {r['seed']} | {r['trainer_seconds']:.3f} | {r['environment_transitions']:,} | {r['development_successes']}/128 | {conf} |")
lines += ['', '## Source records and updates','',
 '[Exact run data and source digests](experiment-runs.json). Rebuild with `python scripts/build_experiment_ledger.py` from this checkout and its retained local archives. The generator reads recorded aggregates only; it does not train, evaluate, deserialize native checkpoints or change cloud resources.',
 '', 'Detailed reports: [official StackCube baseline](stackcube-results-20261026.md), [early sweeps](stackcube-overnight-results-20261027.md), [full-horizon investigation](stackcube-settling-20261027.md), [CAPG](stackcube-capg-results-20261027.md), [SAC](stackcube-sac-results-20261027.md), [50M screen](stackcube-precision-results-20261027.md), [50M replication](stackcube-precision-replication-results-20261028.md), [local BC/prior checks](stackcube-amp-local-dataflow-20261027.md), and [residual exploration hypothesis](research-discovery-bc-residual-exploration-v2-20261028.md).', '']
(ROOT/'docs/experiment-ledger.md').write_text('\n'.join(lines))
print(json.dumps(dict(training_runs=len(rows),families=len(groups),trainer_hours=machine['total_new_training_seconds']/3600,document='docs/experiment-ledger.md')))
