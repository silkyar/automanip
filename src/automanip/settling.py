"""Three preregistered settling hypotheses; evaluator and task remain unchanged."""
import hashlib
import inspect
import re

from automanip.stackcube import OFFICIAL_PPO_SHA256


def campaign():
    common=dict(ent_coef=.003,learning_rate=.0003,update_epochs=8,max_training_seconds=1320)
    return dict(campaign_id='stackcube-settling-time-20261027',
                hypothesis='At most 22 trainer minutes and 25M transitions per run: paired control, full-horizon training, late entropy decay, and discount/entropy interaction.',
                requested_transitions=25_000_000,training_seeds=[1,2,3],candidates=[
                    dict(id='settle-timed-control',gamma=.9,partial_reset=True,entropy_schedule='constant',**common),
                    dict(id='settle-full-horizon',gamma=.9,partial_reset=False,entropy_schedule='constant',**common),
                    dict(id='settle-entropy-decay',gamma=.9,partial_reset=True,entropy_schedule='decay-after-15m',**common),
                    dict(id='settle-gamma0925-entropy003',gamma=.925,partial_reset=True,entropy_schedule='constant',**common)])


def upgrade_legacy(value):
    """Explicitly amend the queued pre-time-limit campaign; never accept arbitrary edits."""
    legacy=campaign()
    legacy['campaign_id']='stackcube-settling-20261027'
    legacy['hypothesis']='Test full-horizon training, late entropy decay, and discount/entropy interaction against the 83.6% development reference.'
    legacy['candidates']=[{k:v for k,v in c.items() if k!='max_training_seconds'} for c in legacy['candidates'][1:]]
    if value!=legacy:raise ValueError('cannot amend an unknown legacy campaign')
    return campaign()


def validate_recipe(candidate):
    if type(candidate) is not dict or candidate not in campaign()['candidates']:
        raise ValueError('unregistered settling recipe')
    for key in ('gamma','ent_coef','learning_rate'):
        if type(candidate[key]) is not float:raise ValueError('floating point parameter required')
    if (type(candidate['partial_reset']) is not bool or type(candidate['update_epochs']) is not int or
        type(candidate['max_training_seconds']) is not int):
        raise ValueError('invalid reset/epoch parameter type')
    return candidate


def validate_settling(value):
    expected=campaign()
    if value['candidates']!=expected['candidates'] or value['requested_transitions']!=25_000_000:
        raise ValueError('settling campaign differs from preregistration')
    for candidate in value['candidates']:validate_recipe(candidate)
    return value


def entropy_coefficient(step, total):
    """Stay at 0.003 through 15M transitions, then linearly reach zero."""
    if total<=15_000_000:return .003
    return .003*max(0.,min(1.,(total-step)/(total-15_000_000)))


def trainer_bytes(official, candidate):
    validate_recipe(candidate)
    if hashlib.sha256(official).hexdigest()!=OFFICIAL_PPO_SHA256:
        raise ValueError('official PPO source digest changed')
    source=official.decode()
    entry='if __name__ == "__main__":\n'
    loop='    for iteration in range(1, args.num_iterations + 1):\n'
    if source.count(entry)!=1 or source.count(loop)!=1:raise ValueError('PPO patch anchors changed')
    def block(label,body,indent=''):
        return indent+'# BEGIN AUTOMANIP '+label+'\n'+body+indent+'# END AUTOMANIP '+label+'\n'
    source=source.replace(entry,block('schedule-function',inspect.getsource(entropy_coefficient)+'\n')+entry)
    source=source.replace('    start_time = time.time()\n','    start_time = time.time()\n'+block('deadline',
        '    automanip_deadline = float(os.environ.get("AUTOMANIP_TRAINING_DEADLINE", time.monotonic() + 1300))\n','    '))
    body='        if time.monotonic() >= automanip_deadline:\n            print("TRAINING_TIME_BUDGET_REACHED", global_step, flush=True)\n            break\n'
    if candidate['entropy_schedule']!='constant':
        body+=('        args.ent_coef = entropy_coefficient(global_step + args.batch_size, 25_000_000)\n'
               '        if logger is not None:\n'
               '            logger.add_scalar("charts/training_entropy_coef", args.ent_coef, global_step + args.batch_size)\n')
    source=source.replace(loop,loop+block('loop-budget-schedule',body,'        '))
    anchor='        rollout_time = time.time() - rollout_time\n'
    source=source.replace(anchor,anchor+block('action-audit',
        '        logger.add_scalar("audit/sampled_action_clip_fraction", (actions.abs() > 1).float().mean().item(), global_step)\n'
        '        for automanip_dim in range(actions.shape[-1]):\n'
        '            logger.add_scalar(f"audit/action_clip_dim_{automanip_dim}", (actions[..., automanip_dim].abs() > 1).float().mean().item(), global_step)\n','        '))
    anchor='                loss.backward()\n'
    source=source.replace(anchor,anchor+block('gradient-audit',
        '                if epoch == 0 and start == 0:\n'
        '                    automanip_actor = list(agent.actor_mean.parameters()) + [agent.actor_logstd]\n'
        '                    automanip_ag = sum(p.grad.detach().square().sum() for p in automanip_actor if p.grad is not None).sqrt().item()\n'
        '                    automanip_cg = sum(p.grad.detach().square().sum() for p in agent.critic.parameters() if p.grad is not None).sqrt().item()\n'
        '                    logger.add_scalar("audit/actor_gradient_norm", automanip_ag, global_step)\n'
        '                    logger.add_scalar("audit/critic_gradient_norm", automanip_cg, global_step)\n'
        '                    logger.add_scalar("audit/global_clip_multiplier", min(1., args.max_grad_norm / (np.hypot(automanip_ag, automanip_cg) + 1e-6)), global_step)\n','                '))
    # Assert that all additions are removable, with the entire pinned program preserved.
    if strip_instrumentation(source).encode()!=official:raise ValueError('unexpected change outside training additions')
    return source.encode()


def strip_instrumentation(source):
    return re.sub(r'^[ \t]*# BEGIN AUTOMANIP ([\w-]+)\n.*?^[ \t]*# END AUTOMANIP \1\n','',source,flags=re.M|re.S)
