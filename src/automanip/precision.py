"""Bounded longer-training controls and isolated late-refinement hypotheses."""
import hashlib
import inspect
import re

from automanip.capg import build_trainer as capg_trainer

METHODS = ('precision-constant', 'precision-entropy-anneal', 'precision-lr-anneal')
REQUESTED_STEPS = 50_000_000
MAX_STEPS = 49_971_200


def refinement_schedule(step, method):
    fraction = max(0., min(1., (step - 25_600_000) / (49_971_200 - 25_600_000)))
    if method == 'precision-constant':
        return .003, .0003
    if method == 'precision-entropy-anneal':
        return .003 * (1 - fraction), .0003
    if method == 'precision-lr-anneal':
        return .003, .0003 * (1 - .9 * fraction)
    raise ValueError('unregistered precision recipe')


def strip_precision(source):
    return re.sub(r'^[ \t]*# BEGIN PRECISION ([\w-]+)\n.*?^[ \t]*# END PRECISION \1\n',
                  '', source, flags=re.M | re.S)


def build_trainer(official, method):
    if method not in METHODS:
        raise ValueError('unregistered precision recipe')
    base = capg_trainer(official, 'full-horizon-control').decode()
    source = base

    def insert(anchor, body, label, indent=''):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError('precision patch anchor changed: ' + label)
        addition = (indent + '# BEGIN PRECISION ' + label + '\n' + body
                    + indent + '# END PRECISION ' + label + '\n')
        source = source.replace(anchor, addition + anchor)

    insert('if __name__ == "__main__":\n', inspect.getsource(refinement_schedule) + '\n', 'schedule')
    insert('    for iteration in range(1, args.num_iterations + 1):\n',
           '    automanip_optimizer_steps = 0\n', 'counter-init', '    ')
    insert('        print(f"Epoch: {iteration}, global_step={global_step}")\n',
           f'        args.ent_coef, automanip_lr = refinement_schedule(global_step + args.batch_size, {method!r})\n'
           '        for automanip_group in optimizer.param_groups:\n'
           '            automanip_group["lr"] = automanip_lr\n', 'apply-schedule', '        ')
    # This increment executes only when the optimizer actually steps, including
    # when KL stops an epoch/minibatch early. It is not an inferred epoch count.
    anchor = '                optimizer.step()\n'
    if source.count(anchor) != 1:
        raise ValueError('optimizer anchor changed')
    source = source.replace(anchor, anchor
        + '                # BEGIN PRECISION counter-step\n'
        + '                automanip_optimizer_steps += 1\n'
        + '                # END PRECISION counter-step\n')
    insert('        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)\n',
           '        logger.add_scalar("audit/optimizer_steps", automanip_optimizer_steps, global_step)\n'
           '        logger.add_scalar("audit/training_entropy_coef", args.ent_coef, global_step)\n'
           '        for automanip_dim in range(agent.actor_logstd.shape[1]):\n'
           '            logger.add_scalar(f"audit/logstd_dim_{automanip_dim}", agent.actor_logstd[0, automanip_dim].detach().item(), global_step)\n',
           'accounting', '        ')
    if strip_precision(source) != base:
        raise ValueError('unexpected edit outside precision additions')
    compile(source, '<precision-trainer>', 'exec')
    return source.encode()


def provenance(official):
    return {method:hashlib.sha256(build_trainer(official, method)).hexdigest() for method in METHODS}
