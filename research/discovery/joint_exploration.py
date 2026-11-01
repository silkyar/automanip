"""Behavior-only mixtures; SAC's target policy/learning code stays unchanged."""
import torch


class Collector:
    def __init__(self, mode, seed, device, epsilon=.1):
        if mode not in ('joint', 'coordinate') or epsilon not in (0., .1):
            raise ValueError('Only registered modes; epsilon0 is CPU parity control')
        self.mode, self.epsilon = mode, epsilon
        self.generator = torch.Generator(device=device).manual_seed(1145329+seed)
        self.work = dict(policy_proposal_rows=0, coin_scalar_draws=0, uniform_scalar_draws=0,
                         rows_with_any_uniform=0, rows_with_all_uniform=0, uniform_per_coordinate=[0]*8)

    def sample(self, actor, observations):
        # Always consume the original policy's global RNG exactly once, including
        # when the independent collector replaces a complete action vector.
        proposal, proposal_logprob = actor.sample(observations)
        coins = torch.rand(proposal.shape, generator=self.generator, device=proposal.device)
        uniform = 2*torch.rand(proposal.shape, generator=self.generator,
                              device=proposal.device, dtype=proposal.dtype)-1
        mask = coins < self.epsilon
        if self.mode == 'joint': mask = mask[:, :1].expand_as(proposal)
        action = torch.where(mask, uniform, proposal)
        counts = mask.sum(0).tolist()
        self.work['policy_proposal_rows'] += len(proposal)
        self.work['coin_scalar_draws'] += proposal.numel()
        self.work['uniform_scalar_draws'] += proposal.numel()
        self.work['rows_with_any_uniform'] += int(mask.any(1).sum())
        self.work['rows_with_all_uniform'] += int(mask.all(1).sum())
        self.work['uniform_per_coordinate'] = [a+int(b) for a, b in zip(self.work['uniform_per_coordinate'], counts)]
        # This density describes the unmodified proposal, NOT the executed
        # mixture action. Replay/actor/critic updates never consume this field.
        return action, proposal_logprob, proposal, mask
