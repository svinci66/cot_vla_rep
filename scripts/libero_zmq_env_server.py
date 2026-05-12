#!/usr/bin/env python
"""LIBERO online rollout server that queries an external model process over ZMQ.

Run this in the LIBERO conda environment. It owns env.reset(), env.step(),
rendering, and success-rate accounting. A separate model process receives
uint8 observations and returns 7-DoF actions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

m.patch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online LIBERO rollout via ZMQ model worker.")
    parser.add_argument("--suite", default="libero_goal", choices=("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"))
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--camera", default="agentview_image")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state-offset", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument("--request-timeout-ms", type=int, default=120000)
    parser.add_argument("--save-failures", action="store_true", help="Save per-episode final info even for failures.")
    return parser.parse_args()


def get_task_bddl_file(task: Any) -> str:
    from libero.libero.utils import get_libero_path

    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def pack(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, default=m.encode, use_bin_type=True)


def unpack(payload: bytes) -> dict[str, Any]:
    return msgpack.unpackb(payload, object_hook=m.decode, raw=False)


def obs_to_payload(obs: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float64:
                value = value.astype(np.float32)
            payload[key] = value
    return payload


def is_success(env, reward: float, done: bool, info: dict[str, Any]) -> bool:
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                return bool(info[key])
    if hasattr(env, "check_success"):
        try:
            return bool(env.check_success())
        except Exception:
            pass
    return bool(done and reward > 0)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task = task_suite.get_task(args.task_id)
    instruction = task.language
    bddl_file = get_task_bddl_file(task)
    init_states = task_suite.get_task_init_states(args.task_id)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": args.height,
        "camera_widths": args.width,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.request_timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.request_timeout_ms)
    endpoint = f"tcp://{args.host}:{args.port}"
    socket.connect(endpoint)

    print("=" * 80)
    print("LIBERO ZMQ Online Rollout")
    print("=" * 80)
    print(f"endpoint: {endpoint}")
    print(f"suite/task: {args.suite}/{args.task_id}")
    print(f"task_name: {task.name}")
    print(f"instruction: {instruction}")
    print(f"episodes: {args.episodes}, max_steps: {args.max_steps}")
    print(f"bddl_file: {bddl_file}")

    # Lightweight readiness handshake.
    socket.send(pack({"type": "hello", "suite": args.suite, "task_id": args.task_id}))
    hello = unpack(socket.recv())
    print(f"model_worker: {hello}")

    results = []
    success_count = 0
    try:
        for episode_idx in range(args.episodes):
            obs = env.reset()
            init_state_id = (args.init_state_offset + episode_idx) % len(init_states)
            obs = env.set_init_state(init_states[init_state_id])
            episode_reward = 0.0
            success = False
            final_info = {}
            start_time = time.time()

            for step_idx in range(args.max_steps):
                request = {
                    "type": "act",
                    "episode": episode_idx,
                    "step": step_idx,
                    "instruction": instruction,
                    "camera": args.camera,
                    "obs": obs_to_payload(obs),
                }
                socket.send(pack(request))
                response = unpack(socket.recv())
                if "error" in response:
                    raise RuntimeError(f"model worker error: {response['error']}")

                action = np.asarray(response["action"], dtype=np.float32).reshape(-1)
                if action.shape[0] != 7:
                    raise ValueError(f"Expected action shape (7,), got {action.shape}")
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

                obs, reward, done, info = env.step(action)
                reward = float(reward)
                episode_reward += reward
                final_info = dict(info) if isinstance(info, dict) else {}
                success = is_success(env, reward, bool(done), final_info)
                if success or done:
                    break

            elapsed = time.time() - start_time
            success_count += int(success)
            episode_result = {
                "episode": episode_idx,
                "init_state_id": init_state_id,
                "success": bool(success),
                "steps": step_idx + 1,
                "reward": episode_reward,
                "elapsed_sec": elapsed,
            }
            if args.save_failures or success:
                episode_result["final_info"] = final_info
            results.append(episode_result)
            print(
                f"episode={episode_idx + 1}/{args.episodes} "
                f"success={success} steps={step_idx + 1} "
                f"reward={episode_reward:.3f} elapsed={elapsed:.1f}s"
            )
    finally:
        try:
            socket.send(pack({"type": "close"}))
            _ = socket.recv()
        except Exception:
            pass
        socket.close(0)
        env.close()

    summary = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": instruction,
        "episodes": args.episodes,
        "success_count": success_count,
        "success_rate": success_count / max(args.episodes, 1),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "results": results,
    }
    print("=" * 80)
    print(f"success_rate = {summary['success_rate'] * 100:.2f}% ({success_count}/{args.episodes})")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
