"""Generate two narrow from-scratch PPO programs from the pinned strong recipe.

Shared instrumentation exports safe actors and counts real work. The sole
learning intervention replaces the complete GAE target block by finite MC.
No simulator, training or cloud work occurs in this preparation script.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/discovery/upstream/precision-constant-base.py'
BASE_SHA = 'cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082'

MC_BLOCK = '''        # bootstrap value according to termination and truncation
        with torch.no_grad():
            if (dones[1:] != 0).any() or not (next_done == 1).all():
                raise ValueError('Finite MC requires complete aligned50-step episodes')
            returns = torch.empty_like(rewards)
            future = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                future = rewards[t] + args.gamma * future
                returns[t] = future
            advantages = returns - values

'''

SHARED_INIT = '''    # Shared data-only instrumentation; never consumes random numbers.
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from export_helper import export_actor
    automanip_root = Path('runs')/run_name
    automanip_owner = envs._env.unwrapped
    if envs.single_observation_space.shape != (48,) or envs.single_action_space.shape != (8,) or max_episode_steps != 50:
        raise ValueError('Frozen StackCube interface/horizon differs')
    if automanip_owner.control_freq != 20 or automanip_owner.sim_freq != 100:
        raise ValueError('Physical clocks differ')
    (automanip_root/'environment.json').write_text(json.dumps(dict(reward_mode=automanip_owner.reward_mode,control_freq=automanip_owner.control_freq,sim_freq=automanip_owner.sim_freq,horizon=max_episode_steps)))
    automanip_journal = (automanip_root/'work.jsonl').open('x',buffering=1)
    automanip_checkpoints = (0,512000,1024000,2048000,4096000,6144000,8192000,10240000)
    def automanip_event(kind,count):
        automanip_journal.write(json.dumps(dict(event=kind,count=int(count)))+'\\n')
    def automanip_export():
        artifact = export_actor(SimpleNamespace(net=agent.actor_mean,hidden_sizes=[256,256,256]),automanip_root/'actors'/str(global_step))
        automanip_journal.write(json.dumps(dict(event='actor_export',steps=global_step,actor=artifact,trainer_seconds=time.monotonic()-automanip_process_started))+'\\n')
    automanip_export()
'''

SHARED_TRACE = '''        # Prespecified first64 lanes, independent of success/failure.
        np.savez_compressed(automanip_root/('trace-'+str(global_step)+'.npz'),
            observations=obs[:,:64].detach().cpu().numpy(),
            raw_actions=actions[:,:64].detach().cpu().numpy(),
            executed_actions=clip_action(actions[:,:64]).cpu().numpy(),
            rewards=rewards[:,:64].detach().cpu().numpy(),
            dones=dones[:,:64].detach().cpu().numpy(),
            values=values[:,:64].detach().cpu().numpy(),
            final_values=final_values[:,:64].detach().cpu().numpy(),
            next_observations=next_obs[:64].detach().cpu().numpy(),
            next_done=next_done[:64].detach().cpu().numpy())
'''


def gae_region(source):
    start = source.index('        # bootstrap value according to termination and truncation\n')
    end = source.index('        # flatten the batch\n', start)
    return start, end


def update_region(source):
    start = source.index('        # Optimizing the policy and value network\n')
    end = source.index('        update_time = time.time() - update_time\n', start)
    return source[start:end]


def build_sources(raw):
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Strong baseline source changed')
    base = raw.decode()
    source = base.replace('import time\n', 'import time\nautomanip_process_started = time.monotonic()\n', 1)
    source = source.replace('    args.batch_size = int(args.num_envs * args.num_steps)',
                            "    if args.checkpoint or args.evaluate or args.partial_reset or args.num_steps != 50 or args.env_id != 'StackCube-v1':\n        raise ValueError('Closed from-scratch StackCube training contract')\n    args.batch_size = int(args.num_envs * args.num_steps)", 1)
    source = source.replace('        if iteration % args.eval_freq == 1:\n', '        if False:  # Independent scoring follows training in another image.\n', 1)
    source = source.replace('    for iteration in range(1, args.num_iterations + 1):\n', SHARED_INIT+'    for iteration in range(1, args.num_iterations + 1):\n', 1)
    needle = '            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))\n'
    assert source.count(needle) == 1
    source = source.replace(needle, "            automanip_event('environment_reserved',global_step)\n"+needle+"            automanip_event('environment_completed',global_step)\n")
    needle = '                optimizer.step()\n'
    source = source.replace(needle, "                automanip_event('optimizer_reserved',automanip_optimizer_steps+1)\n"+needle+"                automanip_event('optimizer_completed',automanip_optimizer_steps+1)\n", 1)
    source = source.replace('        rollout_time = time.time() - rollout_time\n', SHARED_TRACE+'        rollout_time = time.time() - rollout_time\n', 1)
    source = source.replace('        # BEGIN PRECISION accounting\n', "        if global_step in automanip_checkpoints:\n            automanip_export()\n        # BEGIN PRECISION accounting\n", 1)
    source = source.replace('        logger.close()\n', "        if global_step not in automanip_checkpoints:\n            automanip_export()\n        automanip_event('training_complete',global_step)\n        automanip_journal.close()\n        logger.close()\n", 1)
    # Learning/update source remains literal except common work counters.
    if update_region(source).replace("                automanip_event('optimizer_reserved',automanip_optimizer_steps+1)\n", '').replace("                automanip_event('optimizer_completed',automanip_optimizer_steps+1)\n", '') != update_region(base):
        raise ValueError('Unexpected optimizer change')
    lo, hi = gae_region(source)
    candidate = source[:lo]+MC_BLOCK+source[hi:]
    for program in [source, candidate]:
        ast.parse(program)
    return {'ppo-gae': source, 'ppo-finite-mc': candidate}


def prepare(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    sources = build_sources(BASE.read_bytes())
    for name, source in sources.items():
        (output/(name+'.py')).write_text(source)
    # Existing safe exporter only: no student initialization or AMP training.
    raw = (ROOT/'research/discovery/amp_v1.py').read_text()
    tree = ast.parse(raw); function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export_actor')
    helper = 'import torch\nfrom torch import nn\nfrom pathlib import Path\nimport hashlib,json\nTASK_FINGERPRINT="b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f"\n\n'+ast.get_source_segment(raw, function)+'\n'
    (output/'export_helper.py').write_text(helper)
    training_plan = dict(format='automanip-scratch-credit-training-v1',execution_authorized=True,
                         programs={name:hashlib.sha256((output/(name+'.py')).read_bytes()).hexdigest() for name in sources},
                         helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),seeds=[1,2,3],
                         max_training_transitions_per_trial=10240000,trainer_seconds=600,
                         checkpoints=[0,512000,1024000,2048000,4096000,6144000,8192000,10240000],
                         source_sha256=BASE_SHA,source_acquisition_transitions=0,teacher_data=False,
                         observation_dim=48,action_dim=8,gamma=.9,gae_lambda=.9,entropy=.003,
                         num_envs=1024,num_steps=50,num_minibatches=32,update_epochs=8,
                         learning_rate=.0003,partial_reset=False,checkpoint=None,
                         production_pins={'mani_skill':'3.0.0b21','sapien':'3.0.0b1','torch':'2.6.0+cu124','numpy':'1.26.4','gymnasium':'0.29.1'})
    (output/'training-plan.json').write_text(json.dumps(training_plan,indent=2)+'\n')
    # This local scientific protocol is NOT put into the trainer image.
    protocol=dict(format='automanip-scratch-credit-experiment-v1',created_at=time.time(),execution_authorized=True,
                  hypothesis='Full sampled future-reward targets may improve early from-scratch PPO learning versus the tuned GAE baseline; test transfer of the assisted one-step target signal to actual learning, not assume it.',
                  comparison='Same strong precision-constant recipe; shared safe export, blind first64lane trace sampling and work accounting. Only target block changes: GAE(lambda.9,original bootstrap) versus finite50-step MC(gamma.9,zero terminal tail). This is a combined target intervention, not lambda-only or tail-only attribution.',
                  initialization='Fresh independent random initialization seed1/2/3; no teacher, BC policy, demonstrations or old experience',
                  training_plan=training_plan, max_training_transitions=61440000,
                  max_public_development_transitions=307200,
                  scorer='Separate image, unchanged physical StackCube evaluator, safe actor JSON only, candidate volume mounted read-only. No trainer feedback during this registered experiment.',
                  development_seed_bases=[94100000,94101000,94102000],episodes_per_checkpoint=128,
                  primary='Paired per-seed success AUC over common completed prespecified transition checkpoints; report all3seeds and point curves. T80/T90 require two consecutive checkpoints over threshold; no interpolation and censor unreached crossings.',
                  material_result='At least2x earlier T80 across all3seed pairs without reduced final success, with no greater measured total learner compute, would justify replication. 10x remains the original objective; lesser gains must be labeled accurately. If crossings censored or seeds disagree, do not assert resolved efficiency gain.',
                  wall_time='Same600s trainer process cap and10.24M interaction ceiling. Report achieved work, optimizer updates, collection/learner/export overhead separately; equal ceilings do not guarantee equal consumed compute.',
                  final_queries=0,confirmation_queries=0,followup='Reassess after learning curves. No automatic optimizer/architecture sweep or extension of ceilings.')
    (output.parent/'scientific-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(programs=training_plan['programs'],helper_sha256=training_plan['helper_sha256'],new_training_launched=False)))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
