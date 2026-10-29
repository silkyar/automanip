"""Training-only clipped-action likelihood experiment, with unchanged actor export.

Fujita & Maeda, ICML 2018: https://proceedings.mlr.press/v80/fujita18a.html
This is an adaptation for PPO, not an established improvement on StackCube.
"""
import hashlib
import inspect

from automanip.settling import campaign, trainer_bytes


def clipped_gaussian_log_prob(distribution, action):
    """Log density inside (-1,1), log boundary probability mass at either limit."""
    import torch
    mean,std=distribution.loc,distribution.scale
    interior=distribution.log_prob(action.clamp(-1.,1.))
    lower=torch.special.log_ndtr((-1.-mean)/std)
    upper=torch.special.log_ndtr((mean-1.)/std)
    return torch.where(action<=-1.,lower,torch.where(action>=1.,upper,interior))


def experiment():
    return dict(experiment_id='stackcube-capg-20261027',
                hypothesis='Conditioning the PPO score on the executed clipped action reduces wasted gradient variance when Gaussian samples saturate.',
                reference='full-horizon gamma0.9 entropy0.003',training_seeds=[1,2,3],
                max_training_seconds=1320,requested_transition_ceiling=25_000_000,
                methods=['full-horizon-control','full-horizon-capg'],
                unchanged=['actor and critic architecture','Gaussian sampling','latent Gaussian entropy regularizer',
                           'optimizer','task and reward','observation/action interface','frozen evaluator'],
                confirmation_queries=0,promoted=False,
                note='CAPG changes rollout/update log probabilities and hence PPO ratios; it does not change the actions executed at fixed weights. No StackCube performance claim yet.')


def build_trainer(official, method):
    if method not in experiment()['methods']:raise ValueError('unregistered CAPG method')
    reference=next(c for c in campaign()['candidates'] if c['id']=='settle-full-horizon')
    base=trainer_bytes(official,reference)
    if method=='full-horizon-control':return base
    source=base.decode();entry='if __name__ == "__main__":\n'
    old='        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)\n'
    new='        return action, clipped_gaussian_log_prob(probs, action).sum(1), probs.entropy().sum(1), self.critic(x)\n'
    if source.count(entry)!=1 or source.count(old)!=1:raise ValueError('PPO likelihood patch anchor changed')
    block='# BEGIN CAPG FUNCTION\n'+inspect.getsource(clipped_gaussian_log_prob)+'\n# END CAPG FUNCTION\n'
    result=source.replace(entry,block+entry).replace(old,new)
    if result.replace(block,'').replace(new,old)!=source:raise ValueError('CAPG patch changed unrelated training code')
    return result.encode()


def provenance(official):
    return {method:hashlib.sha256(build_trainer(official,method)).hexdigest() for method in experiment()['methods']}
