"""Matched from-scratch PPO: actor KL stop versus critic completion within8 epochs.

Known policy/value optimization separation, related to PPG, not full PPG.
No extra epochs, teacher inputs, reward changes or new actor/critic parameters.
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

BASE_SHA256 = "cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082"
METHODS = {"scratch-kl-stop-control": False, "scratch-kl-stop-critic": True}


def critic_only_minibatch(agent, optimizer, observations, targets, value_coefficient,
                          max_grad_norm, deadline):
    started = time.monotonic()
    # Even a skipped deadline branch clears stale actor gradients; no Adam step
    # is performed, and subsequent critic-only branches keep them absent.
    optimizer.zero_grad(set_to_none=True)
    if time.monotonic() >= deadline:
        return dict(optimizer_calls=0, rows=0, seconds=time.monotonic()-started, value_loss=None)
    actor = list(agent.actor_mean.parameters())+[agent.actor_logstd]
    critic = list(agent.critic.parameters())
    if {id(p) for p in actor} & {id(p) for p in critic}:
        raise ValueError("Independent public actor/critic parameters required")
    # Setting gradients to zeros permits stale Adam momentum to move the actor.
    value_loss = .5*(agent.critic(observations).flatten()-targets.detach()).square().mean()
    loss = value_coefficient*value_loss
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite value-only loss")
    loss.backward()
    if any(parameter.grad is not None for parameter in actor):
        raise RuntimeError("Actor gradient entered the critic-only branch")
    torch.nn.utils.clip_grad_norm_(critic, max_grad_norm)
    optimizer.step()
    measured_loss = float(value_loss.detach())
    return dict(optimizer_calls=1, rows=len(observations), seconds=time.monotonic()-started,
                value_loss=measured_loss)


def strip_additions(source):
    return re.sub(r"^[ \t]*# BEGIN KL_VALUE ([\w-]+)\n.*?^[ \t]*# END KL_VALUE \1\n", "", source, flags=re.M | re.S)


def build_payload(base, method):
    if hashlib.sha256(base).hexdigest() != BASE_SHA256 or method not in METHODS:
        raise ValueError("Reviewed from-scratch PPO base or method differs")
    source = base.decode()
    def insert(anchor, body, label, indent=""):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError("Public source anchor changed: "+label)
        source = source.replace(anchor, anchor+indent+"# BEGIN KL_VALUE "+label+"\n"+body+indent+"# END KL_VALUE "+label+"\n")
    helper = ("import json\n"
              f"AUTOMANIP_CONTINUE_CRITIC = {METHODS[method]!r}\n"
              f"AUTOMANIP_KL_VALUE_METHOD = {method!r}\n"
              f"AUTOMANIP_KL_VALUE_BASE_SHA256 = {BASE_SHA256!r}\n"
              f"AUTOMANIP_KL_VALUE_PATCH_SHA256 = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}\n"
              +inspect.getsource(critic_only_minibatch)+"\n")
    insert("# END PRECISION schedule\n", helper, "helper")
    configuration = """    if args.checkpoint is not None:
        raise ValueError("From-scratch track forbids checkpoint/teacher initialization")
    if args.env_id != "StackCube-v1" or args.control_mode != "pd_joint_delta_pos" or args.num_steps != 50 or args.partial_reset or args.clip_vloss:
        raise ValueError("Fixed full-horizon state task and original value MSE required")
    if args.update_epochs != 8:
        raise ValueError("Matched KL-value pilot requires exactly8 configured update epochs")
"""
    insert("    args = tyro.cli(Args)\n", configuration, "configuration", "    ")
    insert("    # END PRECISION counter-init\n",
           "    automanip_cumulative_critic_only_calls = 0\n    automanip_cumulative_critic_only_rows = 0\n", "totals", "    ")
    initialize = """        automanip_actor_stopped = False
        automanip_kl_stop_epoch = None
        automanip_kl_stop_minibatch = None
        automanip_numpy_state_at_stop = None
        automanip_joint_calls = 0
        automanip_critic_only_calls = 0
        automanip_critic_only_rows = 0
        automanip_critic_only_seconds = 0.
        automanip_critic_only_loss_sum = 0.
        automanip_critic_only_deadline_skips = 0
        automanip_update_loop_started = time.monotonic()
"""
    insert("        clipfracs = []\n", initialize, "rollout-counters", "        ")
    extra_body = """                    automanip_only = critic_only_minibatch(agent, optimizer, b_obs[mb_inds], b_returns[mb_inds],
                        args.vf_coef, args.max_grad_norm, automanip_deadline)
                    automanip_critic_only_calls += automanip_only["optimizer_calls"]
                    automanip_critic_only_rows += automanip_only["rows"]
                    automanip_critic_only_seconds += automanip_only["seconds"]
                    if automanip_only["value_loss"] is not None:
                        automanip_critic_only_loss_sum += automanip_only["value_loss"]*automanip_only["rows"]
                    else:
                        automanip_critic_only_deadline_skips += 1
                    continue
"""
    insert("                mb_inds = b_inds[start:end]\n",
           "                if AUTOMANIP_CONTINUE_CRITIC and automanip_actor_stopped:\n"+extra_body,
           "already-stopped", "                ")
    stop = """                    automanip_actor_stopped = True
                    automanip_kl_stop_epoch = epoch
                    automanip_kl_stop_minibatch = start//args.minibatch_size
                    automanip_numpy_state_at_stop = np.random.get_state()
                    if AUTOMANIP_CONTINUE_CRITIC:
"""
    # Same minibatch was skipped by the control when KL exceeded the threshold.
    # In the candidate only the original value objective runs on that minibatch.
    stop += "\n".join("    "+line if line else line for line in extra_body.split("\n"))
    insert("                if args.target_kl is not None and approx_kl > args.target_kl:\n", stop,
           "first-stop", "                    ")
    insert("                # END PRECISION counter-step\n", "                automanip_joint_calls += 1\n", "joint-count", "                ")
    insert("\n            if args.target_kl is not None and approx_kl > args.target_kl:\n",
           "                if AUTOMANIP_CONTINUE_CRITIC and automanip_actor_stopped:\n                    continue\n",
           "remaining-epochs", "                ")
    finish = """        if automanip_numpy_state_at_stop is not None:
            # Critic completion must not consume future policy minibatch RNG.
            np.random.set_state(automanip_numpy_state_at_stop)
        automanip_cumulative_critic_only_calls += automanip_critic_only_calls
        automanip_cumulative_critic_only_rows += automanip_critic_only_rows
        automanip_kl_value_report = dict(
            method=AUTOMANIP_KL_VALUE_METHOD, base_source_sha256=AUTOMANIP_KL_VALUE_BASE_SHA256,
            patch_source_sha256=AUTOMANIP_KL_VALUE_PATCH_SHA256,
            new_environment_transitions=global_step, configured_epochs=args.update_epochs,
            reached_epoch_count=epoch+1, kl_stopped=automanip_actor_stopped,
            kl_stop_epoch_0based=automanip_kl_stop_epoch, kl_stop_minibatch_0based=automanip_kl_stop_minibatch,
            observed_kl=float(approx_kl), joint_actor_critic_optimizer_calls=automanip_joint_calls,
            actor_optimizer_calls=automanip_joint_calls,
            critic_only_optimizer_calls=automanip_critic_only_calls,
            critic_optimizer_calls=automanip_joint_calls+automanip_critic_only_calls,
            optimizer_calls_total=automanip_joint_calls+automanip_critic_only_calls,
            cumulative_joint_actor_critic_calls=automanip_optimizer_steps,
            cumulative_actor_optimizer_calls=automanip_optimizer_steps,
            cumulative_critic_only_optimizer_calls=automanip_cumulative_critic_only_calls,
            cumulative_critic_optimizer_calls=automanip_optimizer_steps+automanip_cumulative_critic_only_calls,
            cumulative_optimizer_calls_total=automanip_optimizer_steps+automanip_cumulative_critic_only_calls,
            critic_only_training_rows=automanip_critic_only_rows,
            cumulative_critic_only_training_rows=automanip_cumulative_critic_only_rows,
            critic_only_seconds=automanip_critic_only_seconds,
            full_optimizer_loop_seconds=time.monotonic()-automanip_update_loop_started,
            critic_only_deadline_skips=automanip_critic_only_deadline_skips,
            critic_only_mean_value_loss=(automanip_critic_only_loss_sum/automanip_critic_only_rows if automanip_critic_only_rows else None),
            actor_updates_after_kl_stop=0, critic_only_actor_gradients_none=True,
            target_refreshes_in_critic_only_phase=0, new_models_or_teacher_data=False)
        for automanip_key, automanip_value in automanip_kl_value_report.items():
            if isinstance(automanip_value, (int, float)) and not isinstance(automanip_value, bool):
                logger.add_scalar("audit/kl_value/"+automanip_key, automanip_value, global_step)
        print("AUTOMANIP_KL_VALUE_UPDATES "+json.dumps(automanip_kl_value_report, sort_keys=True), flush=True)
"""
    insert("        update_time = time.time() - update_time\n", finish, "receipt", "        ")
    if strip_additions(source).encode() != base:
        raise ValueError("Unmarked change to reviewed PPO base")
    compile(source, method+"-payload", "exec")
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.base.read_bytes(), args.method)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(payload).hexdigest())))
