"""Minimal, exactly removable score-objective patches to owner's public PPO base.

Runtime payloads are self-contained; no local/protected module imports.
The owner's existing launcher retains responsibility for isolation, checkpoint
staging, continuation accounting, external scoring and the hard trainer lease.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re

from research.discovery.episodic_risk_v1 import (
    actor_score_advantages, episode_returns, score_weights, training_diagnostics,
)

BASE_SHA256 = "cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082"
METHODS = {"risk-ppo-control": "ordinary", "risk-episode-mean": "mean", "risk-episode-cvar10": "cvar"}


def strip_risk_additions(source):
    return re.sub(r"^[ \t]*# BEGIN DISCOVERY_RISK ([\w-]+)\n.*?^[ \t]*# END DISCOVERY_RISK \1\n", "", source, flags=re.M | re.S)


def build_precision_trainer(base, method):
    if hashlib.sha256(base).hexdigest() != BASE_SHA256:
        raise ValueError("Owner-provided public precision base digest changed")
    if method not in METHODS:
        raise ValueError("Unregistered risk pilot method")
    source = base.decode()
    def insert(anchor, body, label, indent=""):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError(f"Public source anchor changed: {label}")
        block = indent+"# BEGIN DISCOVERY_RISK "+label+"\n"+body+indent+"# END DISCOVERY_RISK "+label+"\n"
        source = source.replace(anchor, anchor+block)
    helpers = "import json\nAUTOMANIP_RISK_MODE = "+repr(METHODS[method])+"\n"+"\n\n".join(
        inspect.getsource(fn) for fn in (episode_returns, score_weights, training_diagnostics, actor_score_advantages))+"\n"
    # Insert immediately after the final public schedule block, before main.
    insert("# END PRECISION schedule\n", helpers, "helpers")
    configuration = """    if args.env_id != "StackCube-v1" or args.control_mode != "pd_joint_delta_pos" or args.num_steps != 50 or args.partial_reset:
        raise ValueError("Risk pilot requires the unchanged public full-horizon StackCube interface")
    if args.num_envs < 32:
        raise ValueError("At least32 complete episodes required for tail estimation")
    if AUTOMANIP_RISK_MODE != "ordinary":
        args.norm_adv = False
"""
    insert("    args = tyro.cli(Args)\n", configuration, "configuration", "    ")
    actor = """        automanip_score, automanip_diagnostic = actor_score_advantages(
            rewards, dones, next_done, infos, args.gamma, AUTOMANIP_RISK_MODE)
        if AUTOMANIP_RISK_MODE != "ordinary":
            advantages = automanip_score
        automanip_diagnostic.update(new_transitions=global_step, training_seed=args.seed)
        for automanip_key, automanip_value in automanip_diagnostic.items():
            if isinstance(automanip_value, (int, float)) and not isinstance(automanip_value, bool):
                logger.add_scalar("audit/risk/"+automanip_key, automanip_value, global_step)
        print("AUTOMANIP_RISK_DIAGNOSTIC "+json.dumps(automanip_diagnostic, sort_keys=True), flush=True)
"""
    insert("            returns = advantages + values\n", actor, "actor-score", "        ")
    if strip_risk_additions(source).encode() != base:
        raise ValueError("Unmarked change to public base")
    compile(source, f"{method}-payload", "exec")
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_precision_trainer(args.base.read_bytes(), args.method)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(dict(method=args.method, path=str(args.output), sha256=hashlib.sha256(payload).hexdigest())))
