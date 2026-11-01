"""Protect deterministic executed actions, rather than clipped-away raw values."""
import torch


def objective(delta, teacher_features, teacher_targets, teacher_original,
              rehearsal_features, rehearsal_original, executed):
    prediction = teacher_original+teacher_features@delta
    rehearsal = rehearsal_original+rehearsal_features@delta
    # Keep the teacher term raw: clipping a wrong saturated prediction would
    # remove the gradient required to escape that saturation.
    teacher_loss = (prediction-teacher_targets).square().sum(-1).mean()
    difference = (rehearsal.clamp(-1, 1)-rehearsal_original.clamp(-1, 1)
                  if executed else rehearsal-rehearsal_original)
    memory_loss = difference.square().sum(-1).mean()
    return teacher_loss+4*memory_loss+.001*delta.square().sum()
