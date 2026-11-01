"""Only route the existing adaptive alpha differently into the soft backup."""
import os
import torch
from replay_learner_base import Learner as BaseLearner, Replay, transition_successor, export_actor

ROUTES = {'high-actor-low-backup': (1., .005), 'low-actor-high-backup': (.005, 200.)}


class Learner(BaseLearner):
    def __init__(self, *args, **kwargs):
        route = os.environ['AUTOMANIP_ENTROPY_ROUTE']
        if route not in ROUTES: raise ValueError('Unregistered entropy route')
        initial_alpha, self.backup_multiplier = ROUTES[route]
        super().__init__(*args, **kwargs)
        if self.method != 'target-sac': raise ValueError('Route test requires original target SAC')
        if initial_alpha != 1.:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(.005, device=self.log_alpha.device).log())
        self.route = route

    def critic_target(self, data, next_action, next_logprob, alpha):
        current1, current2, target = super().critic_target(data, next_action, next_logprob, alpha)
        delta = self.gamma*(1-data['stop'])*alpha*(1-self.backup_multiplier)*next_logprob
        return current1, current2, (target+delta).detach()

    def update(self, data):
        stats = super().update(data)
        stats['actor_alpha'] = stats['alpha']
        stats['critic_target_alpha'] = stats['alpha']*self.backup_multiplier
        return stats
