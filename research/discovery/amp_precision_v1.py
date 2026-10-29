"""Exactly marked, self-contained BC+PPO versus BC+PPO+AMP payloads.

Random BC student artifact only; teacher policy weights are not a student input.
The trusted owner supplies isolated data mounts, budgets and external scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re

from research.discovery import amp_v1 as core

BASE_SHA256 = "cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082"
METHODS = {"assist-bc-ppo": False, "assist-bc-ppo-amp": True}


def assist_initialize(agent, args, use_prior, device):
    if args.assist_bc_actor == "" or args.assist_dataset == "" or args.assist_bc_report == "":
        raise ValueError("Explicit public demo dataset and random-student BC artifact required")
    data = load_demos(args.assist_dataset)
    with torch.random.fork_rng(devices=[]):
        actor, manifest = read_actor(args.assist_bc_actor, device)
    if actor.hidden_sizes != [256, 256, 256]:
        raise ValueError("BC artifact must match pinned PPO actor architecture")
    bc_report_raw = Path(args.assist_bc_report).read_bytes()
    bc_report = json.loads(bc_report_raw)
    if (bc_report.get("teacher_weights_loaded") is not False or bc_report.get("student_initialization") != "random"
            or bc_report.get("comparison_eligible") is not True or bc_report.get("dataset_sha256") != data.dataset_sha256
            or bc_report.get("final", {}).get("weights_sha256") != manifest["weights_sha256"]):
        raise ValueError("BC provenance does not match a random-initialized, production-eligible student")
    agent.actor_mean.load_state_dict(actor.net.state_dict(), strict=True)
    prior = None
    if use_prior:
        obs, next_obs = data.training_pairs(device=device)
        prior = MotionPrior(obs, next_obs, args.seed, device=device)
    metadata = dict(
        track="prior-assisted random student: BC then PPO", teacher_weights_loaded=False,
        bc_report_sha256=hashlib.sha256(bc_report_raw).hexdigest(),
        bc_prefix_optimizer_updates=bc_report["optimizer_updates"],
        bc_prefix_demonstration_minibatch_rows=bc_report["demonstration_minibatch_rows"],
        bc_prefix_wall_seconds=bc_report["wall_seconds"],
        dataset_sha256=data.dataset_sha256, bc_student_weights_sha256=manifest["weights_sha256"],
        teacher_prefix_transitions=data.metadata["teacher_prefix_transitions"],
        teacher_generation_transitions=data.metadata["environment_transitions"],
        teacher_prefix_trainer_seconds=data.metadata["teacher_prefix_trainer_seconds"],
        teacher_prefix_dollars=data.metadata["teacher_prefix_dollars"],
        all_teacher_episodes=data.metadata["completed_episodes"],
        failed_teacher_episodes_preserved=data.metadata["failed_completed_episodes"],
        training_demo_episodes=len(data.train_episode_indices), validation_demo_episodes=len(data.validation_episode_indices),
        amp_enabled=use_prior, amp_style_weight=.25 if use_prior else 0.,
        task_reward_weight=1., discriminator_input="existing state48 followed by actual successor state48",
        discriminator_models_in_agent_serialization=False)
    print("AUTOMANIP_ASSISTANCE "+json.dumps(metadata, sort_keys=True), flush=True)
    return prior


def strip_assist_additions(source):
    return re.sub(r"^[ \t]*# BEGIN DISCOVERY_ASSIST ([\w-]+)\n.*?^[ \t]*# END DISCOVERY_ASSIST \1\n", "", source, flags=re.M | re.S)


def build_assist_trainer(base, method):
    if hashlib.sha256(base).hexdigest() != BASE_SHA256 or method not in METHODS:
        raise ValueError("Public PPO base or assisted method mismatch")
    source = base.decode()
    def insert(anchor, body, label, indent=""):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError("Public source anchor changed: "+label)
        block = indent+"# BEGIN DISCOVERY_ASSIST "+label+"\n"+body+indent+"# END DISCOVERY_ASSIST "+label+"\n"
        source = source.replace(anchor, anchor+block)
    helpers = ("import hashlib, json\nfrom pathlib import Path\n"
               f"TASK_FINGERPRINT = {core.TASK_FINGERPRINT!r}\nDATA_FORMAT = {core.DATA_FORMAT!r}\n"
               f"PRODUCTION_PINS = {core.PRODUCTION_PINS!r}\n"
               f"AUTOMANIP_USE_PRIOR = {METHODS[method]!r}\n")
    selected = (core.StudentActor, core.read_actor, core.DemoData, core.load_demos,
                core.actual_successor, core.amp_style_reward, core.mix_training_reward,
                core.TransitionDiscriminator, core.discriminator_loss, core.MotionPrior,
                assist_initialize)
    helpers += "\n\n".join(inspect.getsource(value) for value in selected)+"\n"
    insert("# END PRECISION schedule\n", helpers, "helpers")
    insert("    finite_horizon_gae: bool = False\n",
           '    assist_dataset: str = ""\n    assist_bc_actor: str = ""\n    assist_bc_report: str = ""\n', "arguments", "    ")
    configuration = """    if args.checkpoint is not None:
        raise ValueError("Assisted student cannot load teacher/native policy weights; use its random-initialized BC actor artifact")
    if args.env_id != "StackCube-v1" or args.control_mode != "pd_joint_delta_pos" or args.num_steps != 50 or args.partial_reset:
        raise ValueError("Assisted track requires the unchanged full-horizon public task/interface")
"""
    insert("    args = tyro.cli(Args)\n", configuration, "configuration", "    ")
    insert("    agent = Agent(envs).to(device)\n",
           "    automanip_prior = assist_initialize(agent, args, AUTOMANIP_USE_PRIOR, device)\n", "initialize", "    ")
    insert("        rollout_time = time.time()\n", "        automanip_style_mean = 0.\n", "rollout-counter", "        ")
    reward = """            if automanip_prior is not None:
                automanip_actual_next = actual_successor(next_obs, next_done, infos)
                automanip_style = automanip_prior.reward_and_remember(obs[step], automanip_actual_next)
                rewards[step] = mix_training_reward(rewards[step], automanip_style*args.reward_scale, .25)
                automanip_style_mean += float(automanip_style.mean())/args.num_steps
"""
    insert("            rewards[step] = reward.view(-1) * args.reward_scale\n", reward, "train-only-reward", "            ")
    update = """        if automanip_prior is not None:
            automanip_prior_started = time.time()
            automanip_prior_stats = automanip_prior.update(rounds=32, batch_size=256)
            automanip_prior_stats["discriminator_update_seconds"] = time.time()-automanip_prior_started
            automanip_prior_stats["mean_style_reward"] = automanip_style_mean
            for automanip_key, automanip_value in automanip_prior_stats.items():
                logger.add_scalar("audit/amp/"+automanip_key, automanip_value, global_step)
            print("AUTOMANIP_PRIOR_DIAGNOSTIC "+json.dumps(automanip_prior_stats, sort_keys=True), flush=True)
"""
    insert("        rollout_time = time.time() - rollout_time\n", update, "discriminator-update", "        ")
    if strip_assist_additions(source).encode() != base:
        raise ValueError("Unmarked alteration to public PPO base")
    compile(source, method+"-payload", "exec")
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_assist_trainer(args.base.read_bytes(), args.method)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(dict(method=args.method, path=str(args.output), sha256=hashlib.sha256(payload).hexdigest())))
