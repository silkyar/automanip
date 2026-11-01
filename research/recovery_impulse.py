"""Explicit COM velocity impulse using the pinned, existing GPU data API.

This is a separately registered instantaneous-impulse task, not a replacement
implementation of a finite-duration force or a simulator version change.
"""
import torch

IMPULSE_NEWTON_SECONDS=.08


def apply_impulse(actor, impulses):
    velocity=actor.linear_velocity.clone()
    mass=actor.mass.to(device=velocity.device,dtype=velocity.dtype).reshape(-1,1)
    impulses=torch.as_tensor(impulses,device=velocity.device,dtype=velocity.dtype)
    if impulses.shape!=velocity.shape or velocity.ndim!=2 or velocity.shape[1]!=3:
        raise ValueError('Impulse shape differs')
    if mass.shape!=(len(velocity),1) or not torch.isfinite(mass).all() or (mass<=0).any():
        raise ValueError('Invalid actual body mass')
    if not torch.isfinite(velocity).all() or not torch.isfinite(impulses).all():raise ValueError('Nonfinite impulse/state')
    actor.set_linear_velocity(velocity+impulses/mass)
    if actor.scene.gpu_sim_enabled:
        actor.scene.px.gpu_apply_rigid_dynamic_data()
        actor.scene._gpu_fetch_all()
    return dict(impulse_newton_seconds=impulses.cpu().tolist(),mass_kg=mass.flatten().cpu().tolist())
