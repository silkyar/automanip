"""CPU audit of the installed, public StackCube reward; no simulator import.

This falsifies a missing-release-incentive hypothesis, not a learning mechanism.
Only the pinned public reward method is extracted. No evaluator is accessed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace


SOURCE_SHA256 = "ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440"


def audit(source_path: Path) -> dict:
    import torch

    start = time.monotonic()
    source = source_path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Public StackCube source differs from inspected release")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StackCubeEnv")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_dense_reward")
    namespace = {"torch": torch, "Any": object, "Dict": dict}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    reward_function = namespace["compute_dense_reward"]

    def reward(*, on_stack, grasped, speed, angular_speed, success=False):
        cube = SimpleNamespace(
            pose=SimpleNamespace(p=torch.tensor([[0., 0., .06 if on_stack else .16]])),
            linear_velocity=torch.tensor([[speed, 0., 0.]]),
            angular_velocity=torch.tensor([[angular_speed, 0., 0.]]),
        )
        robot = SimpleNamespace(
            get_qlimits=lambda: torch.tensor([[[0., .04]]]),
            get_qpos=lambda: torch.tensor([[.008, .008]]),
        )
        env = SimpleNamespace(
            cubeA=cube, cubeB=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0., 0., .02]]))),
            agent=SimpleNamespace(robot=robot, tcp=SimpleNamespace(pose=cube.pose)),
            cube_half_size=torch.tensor([.02, .02, .02]), device="cpu",
        )
        info = {"is_cubeA_grasped": torch.tensor([grasped]),
                "is_cubeA_on_cubeB": torch.tensor([on_stack]), "success": torch.tensor([success])}
        return float(reward_function(env, None, None, info)[0] / 8)

    common = dict(on_stack=True, speed=.02, angular_speed=.6)
    values = {
        "on_stack_grasped_moving": reward(**common, grasped=True),
        "on_stack_released_moving": reward(**common, grasped=False),
        "on_stack_released_slower": reward(on_stack=True, grasped=False, speed=.011, angular_speed=.01),
        "on_stack_released_success": reward(on_stack=True, grasped=False, speed=0., angular_speed=0., success=True),
        "off_stack_grasped": reward(on_stack=False, grasped=True, speed=0., angular_speed=0.),
        "off_stack_released": reward(on_stack=False, grasped=False, speed=0., angular_speed=0.),
    }
    checks = {
        "release_is_rewarded_when_on_stack": values["on_stack_released_moving"] > values["on_stack_grasped_moving"],
        "slowing_is_rewarded_when_on_stack": values["on_stack_released_slower"] > values["on_stack_released_moving"],
        "success_override_is_one": values["on_stack_released_success"] == 1.,
        "premature_release_can_reduce_reward": values["off_stack_released"] < values["off_stack_grasped"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(
        public_task_source_sha256=digest, installed_release="mani_skill==3.0.0b21",
        matching_upstream_release_commit="6226567ce326625fa6eacf9aeb3f4008e4339f7e",
        commit_note="Release mapping from earlier public provenance; remote task bytes not fetched in this audit.",
        values=values, checks=checks, audit_wall_seconds=time.monotonic()-start,
        robot_transitions=0, simulator_queries=0, optimizer_updates=0,
        gpu_seconds=0, paid_api_calls=0, dollars_spent=0,
        scope="Six hand-constructed CPU reward inputs; no dynamics or robot learning claim.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    result["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
