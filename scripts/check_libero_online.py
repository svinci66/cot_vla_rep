#!/usr/bin/env python
"""Smoke-test online LIBERO interaction with OffScreenRenderEnv.

This script intentionally does not import VILA-U. It only verifies that a
server can create a LIBERO task environment, reset to a benchmark initial
state, step dummy 7-DoF actions, and read rendered observations.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check online LIBERO reset/step.")
    parser.add_argument(
        "--suite",
        default="libero_goal",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"),
        help="LIBERO task suite name.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task id in the suite.")
    parser.add_argument("--init-state-id", type=int, default=0, help="Benchmark init state id.")
    parser.add_argument("--steps", type=int, default=10, help="Number of dummy env steps.")
    parser.add_argument("--camera", default="agentview_image", help="Observation image key to report.")
    parser.add_argument("--height", type=int, default=256, help="Camera height.")
    parser.add_argument("--width", type=int, default=256, help="Camera width.")
    parser.add_argument(
        "--mujoco-gl",
        default=os.environ.get("MUJOCO_GL", "egl"),
        help="Set MUJOCO_GL before importing LIBERO/robosuite.",
    )
    return parser.parse_args()


def get_task_bddl_file(task: Any) -> str:
    from libero.libero.utils import get_libero_path

    return os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task = task_suite.get_task(args.task_id)
    bddl_file = get_task_bddl_file(task)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": args.height,
        "camera_widths": args.width,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    try:
        obs = env.reset()
        init_states = task_suite.get_task_init_states(args.task_id)
        if args.init_state_id >= len(init_states):
            raise ValueError(
                f"init-state-id {args.init_state_id} out of range; "
                f"task has {len(init_states)} init states"
            )
        obs = env.set_init_state(init_states[args.init_state_id])

        print(f"suite: {args.suite}")
        print(f"task_id: {args.task_id}")
        print(f"task_name: {task.name}")
        print(f"instruction: {task.language}")
        print(f"bddl_file: {bddl_file}")
        print(f"obs_keys: {sorted(obs.keys())}")

        dummy_action = np.zeros(7, dtype=np.float32)
        total_reward = 0.0
        done = False
        for step in range(args.steps):
            obs, reward, done, info = env.step(dummy_action)
            total_reward += float(reward)
            image = obs.get(args.camera)
            image_shape = None if image is None else tuple(image.shape)
            print(
                f"step={step + 1} reward={float(reward):.3f} "
                f"done={bool(done)} image_shape={image_shape} info_keys={sorted(info.keys())}"
            )
            if done:
                break

        if args.camera not in obs:
            raise RuntimeError(f"camera key {args.camera!r} not found in obs")
        print(f"online_libero_check=ok total_reward={total_reward:.3f} done={bool(done)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
