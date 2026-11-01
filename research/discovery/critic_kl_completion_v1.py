"""From-scratch hypothesis: continue value fitting after actor-KL early stop.

One extra value-only epoch on the same frozen rollout targets; no teacher data,
new actor head, target recursion, replay across rollouts or additional simulation.
Related to PPG's separate value optimization, not an implementation of full PPG.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import time

import torch

PUBLIC_BASE_SHA256 = "cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082"


def critic_only_epoch(agent, optimizer, observations, targets, minibatch_size,
                      generator, value_coefficient=.5, max_grad_norm=.5, deadline=None):
    actor = list(agent.actor_mean.parameters())+[agent.actor_logstd]
    critic = list(agent.critic.parameters())
    if {id(p) for p in actor} & {id(p) for p in critic}:
        raise ValueError("Value-only phase requires disjoint public actor/critic parameters")
    if minibatch_size < 1 or targets.shape != (len(observations),) or len(observations) < 1:
        raise ValueError("Invalid fixed rollout value data")
    fixed = targets.detach()
    with torch.no_grad():
        before = float((agent.critic(observations).flatten()-fixed).square().mean())
    order = torch.randperm(len(observations), generator=generator, device=observations.device)
    updates = rows = 0
    started = time.monotonic()
    for start in range(0, len(order), minibatch_size):
        if deadline is not None and time.monotonic() >= deadline:
            break
        index = order[start:start+minibatch_size]
        # Zero tensors would allow stale Adam momentum to move the actor.
        optimizer.zero_grad(set_to_none=True)
        loss = .5*value_coefficient*(agent.critic(observations[index]).flatten()-fixed[index]).square().mean()
        loss.backward()
        if any(parameter.grad is not None for parameter in actor):
            raise RuntimeError("Actor gradient entered a value-only phase")
        torch.nn.utils.clip_grad_norm_(critic, max_grad_norm)
        optimizer.step()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite value-only update")
        updates += 1
        rows += len(index)
    with torch.no_grad():
        after = float((agent.critic(observations).flatten()-fixed).square().mean())
    return dict(extra_critic_optimizer_steps=updates, extra_critic_training_rows=rows,
                extra_critic_forward_rows=rows+2*len(observations),
                fixed_target_mse_before=before, fixed_target_mse_after=after,
                extra_critic_seconds=time.monotonic()-started,
                actor_updates_in_extra_phase=0, new_environment_transitions=0)


def strip_value_additions(source):
    return re.sub(r"^[ \t]*# BEGIN VALUE_COMPLETION ([\w-]+)\n.*?^[ \t]*# END VALUE_COMPLETION \1\n", "", source, flags=re.M | re.S)


def build_payload(base):
    if hashlib.sha256(base).hexdigest() != PUBLIC_BASE_SHA256:
        raise ValueError("Pinned public full-horizon PPO base differs")
    source = base.decode()
    def insert(anchor, body, label, indent=""):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError("Public source anchor changed: "+label)
        source = source.replace(anchor, anchor+indent+"# BEGIN VALUE_COMPLETION "+label+"\n"+body+indent+"# END VALUE_COMPLETION "+label+"\n")
    insert("# END PRECISION schedule\n", "import json\n"+inspect.getsource(critic_only_epoch)+"\n", "helper")
    configuration = """    if args.checkpoint is not None:
        raise ValueError("Value-completion track requires random initialization, no teacher/checkpoint input")
    if args.env_id != "StackCube-v1" or args.control_mode != "pd_joint_delta_pos" or args.num_steps != 50 or args.partial_reset or args.clip_vloss:
        raise ValueError("Fixed public full-horizon state task and ordinary value MSE required")
"""
    insert("    args = tyro.cli(Args)\n", configuration, "configuration", "    ")
    insert("    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)\n",
           "    automanip_value_generator = torch.Generator(device=device).manual_seed(args.seed+51001)\n    automanip_extra_value_steps = 0\n", "generator", "    ")
    extra = """        automanip_value_triggered = args.target_kl is not None and approx_kl > args.target_kl
        if automanip_value_triggered:
            automanip_value_stats = critic_only_epoch(agent, optimizer, b_obs, b_returns,
                args.minibatch_size, automanip_value_generator, args.vf_coef,
                args.max_grad_norm, automanip_deadline)
            automanip_extra_value_steps += automanip_value_stats["extra_critic_optimizer_steps"]
            for automanip_key, automanip_value in automanip_value_stats.items():
                logger.add_scalar("audit/value_only/"+automanip_key, automanip_value, global_step)
            print("AUTOMANIP_VALUE_COMPLETION "+json.dumps(automanip_value_stats, sort_keys=True), flush=True)
        logger.add_scalar("audit/value_only/triggered", int(automanip_value_triggered), global_step)
        logger.add_scalar("audit/value_only/cumulative_steps", automanip_extra_value_steps, global_step)
        logger.add_scalar("audit/actor_optimizer_steps", automanip_optimizer_steps, global_step)
        logger.add_scalar("audit/total_optimizer_steps", automanip_optimizer_steps+automanip_extra_value_steps, global_step)
"""
    insert("        update_time = time.time() - update_time\n", extra, "phase", "        ")
    if strip_value_additions(source).encode() != base:
        raise ValueError("Unexpected change to public PPO base")
    compile(source, "from-scratch-value-completion", "exec")
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.base.read_bytes())
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(payload).hexdigest())))
