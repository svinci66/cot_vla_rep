#!/usr/bin/env python
"""Diagnose HDF5-demo vs online-reset alignment for LIBERO ZMQ eval.

This script assumes a VILA ZMQ model worker is already running. It compares:

- HDF5 demo observation frame used by training
- Online environment observation after resetting to the same demo init state
- Model action chunks for both observations through the same ZMQ worker
- Ground-truth filtered action chunk from ``LiberoGoalDataset``
- Optional replay of filtered demo actions in the online environment
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

from vila_u.utils.libero_action import libero_raw_actions_to_model_actions

m.patch()


def pack(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, default=m.encode, use_bin_type=True)


def unpack(payload: bytes) -> dict[str, Any]:
    return msgpack.unpackb(payload, object_hook=m.decode, raw=False)


def load_demo_init_state(hdf5_path: str, demo_name: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5_file:
        demo = h5_file["data"][demo_name]
        if "init_state" in demo.attrs:
            return np.asarray(demo.attrs["init_state"])
        if "states" in demo and len(demo["states"]) > 0:
            return np.asarray(demo["states"][0])
    raise ValueError(f"No init_state attr or states[0] found for {demo_name!r}")


def non_pause_indices(
    actions: np.ndarray,
    pause_threshold: float = 0.01,
    gripper_pause_threshold: float = 1e-6,
) -> list[int]:
    indices = []
    previous_action = None
    for timestep, action in enumerate(actions):
        position_norm = np.linalg.norm(action[:3])
        rotation_norm = np.linalg.norm(action[3:6])
        if previous_action is not None and action.shape[0] > 6 and previous_action.shape[0] > 6:
            gripper_delta = abs(float(action[6]) - float(previous_action[6]))
        else:
            gripper_delta = abs(float(action[6])) if action.shape[0] > 6 else 0.0
        is_pause = (
            position_norm < pause_threshold
            and rotation_norm < pause_threshold
            and gripper_delta < gripper_pause_threshold
        )
        if not is_pause:
            indices.append(timestep)
        previous_action = action
    return indices


def obs_to_payload(obs: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float64:
                value = value.astype(np.float32)
            payload[key] = value
    return payload


def obs_frame(obs: dict[str, Any], camera: str) -> np.ndarray:
    frame = obs[camera]
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        frame = frame[..., :3]
    elif frame.ndim == 3 and frame.shape[0] >= 3:
        frame = np.moveaxis(frame[:3], 0, -1)
    else:
        raise ValueError(f"Unsupported frame shape for {camera}: {frame.shape}")
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def save_image(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def resize_like(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if image.shape[:2] == target_shape:
        return image
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    resized = pil.resize((target_shape[1], target_shape[0]))
    return np.asarray(resized, dtype=np.uint8)


def frame_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    return {
        "mae": float(diff.mean()),
        "max": float(diff.max()),
        "mse": float(np.square(a - b).mean()),
    }


def action_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    diff = np.abs(pred - gt)
    return {
        "mae": float(diff.mean()),
        "motion_mae": float(diff[:, :6].mean()),
        "first_step_mae": float(diff[0].mean()),
        "per_dim_mae": diff.mean(axis=0).astype(float).tolist(),
    }


def request_chunk(socket, episode: int, instruction: str, camera: str, obs: dict[str, Any]) -> np.ndarray:
    socket.send(pack({"type": "act", "episode": episode, "step": 0, "instruction": instruction, "camera": camera, "obs": obs_to_payload(obs)}))
    response = unpack(socket.recv())
    if "error" in response:
        raise RuntimeError(response.get("traceback") or response["error"])
    chunk_id = response.get("chunk_id")
    first_action = np.asarray(response["action"], dtype=np.float32).reshape(1, -1)
    actions = [first_action[0]]
    for step in range(1, 10):
        socket.send(pack({"type": "act", "episode": episode, "step": step, "instruction": instruction, "camera": camera, "obs": obs_to_payload(obs)}))
        response = unpack(socket.recv())
        if "error" in response:
            raise RuntimeError(response.get("traceback") or response["error"])
        if response.get("chunk_id") != chunk_id:
            raise RuntimeError("Worker started a new chunk before 10 actions were consumed")
        actions.append(np.asarray(response["action"], dtype=np.float32).reshape(-1))
    return np.asarray(actions, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose LIBERO HDF5/online alignment.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--demo-name", default="demo_0")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--camera", default="agentview_image")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-gt-actions", action="store_true")
    parser.add_argument("--replay-steps", type=int, default=120)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = Path(args.data_root) / args.task_file

    with h5py.File(hdf5_path, "r") as h5_file:
        instruction = json.loads(h5_file["data"].attrs["problem_info"])["language_instruction"]
        demo = h5_file["data"][args.demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        filtered_indices = non_pause_indices(actions)
        if len(filtered_indices) < 10:
            raise ValueError(f"Demo {args.demo_name!r} has only {len(filtered_indices)} non-pause actions")
        filtered_timestep = 0
        original_timestep = filtered_indices[filtered_timestep]
        hdf5_frame = np.asarray(demo["obs/agentview_rgb"][original_timestep], dtype=np.uint8)
        gt_indices = filtered_indices[filtered_timestep : filtered_timestep + 10]
        gt_chunk = libero_raw_actions_to_model_actions(demo["actions"][gt_indices])
        replay_actions = np.asarray(demo["actions"][filtered_indices[: args.replay_steps]], dtype=np.float32).clip(-1.0, 1.0)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task = task_suite.get_task(args.task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=args.height, camera_widths=args.width)
    env.seed(args.seed)
    env.reset()
    online_obs = env.set_init_state(load_demo_init_state(str(hdf5_path), args.demo_name))
    online_frame = obs_frame(online_obs, args.camera)

    hdf5_frame_resized = resize_like(hdf5_frame, online_frame.shape[:2])
    hdf5_rot = np.ascontiguousarray(hdf5_frame_resized[::-1, ::-1])
    online_rot = np.ascontiguousarray(online_frame[::-1, ::-1])
    diff = np.abs(hdf5_frame_resized.astype(np.int16) - online_frame.astype(np.int16)).clip(0, 255).astype(np.uint8)

    save_image(output_dir / "hdf5_frame_raw.png", hdf5_frame_resized)
    save_image(output_dir / "hdf5_frame_rot180.png", hdf5_rot)
    save_image(output_dir / "online_frame_raw.png", online_frame)
    save_image(output_dir / "online_frame_rot180.png", online_rot)
    save_image(output_dir / "frame_absdiff_raw.png", diff)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 120000)
    socket.setsockopt(zmq.SNDTIMEO, 120000)
    socket.connect(f"tcp://{args.host}:{args.port}")
    socket.send(pack({"type": "hello", "suite": args.suite, "task_id": args.task_id}))
    hello = unpack(socket.recv())

    hdf5_obs = {args.camera: hdf5_frame_resized}
    pred_hdf5 = request_chunk(socket, episode=100000, instruction=instruction, camera=args.camera, obs=hdf5_obs)
    pred_online = request_chunk(socket, episode=100001, instruction=instruction, camera=args.camera, obs=online_obs)
    socket.close(0)

    replay = {"enabled": bool(args.replay_gt_actions)}
    if args.replay_gt_actions:
        env.reset()
        obs = env.set_init_state(load_demo_init_state(str(hdf5_path), args.demo_name))
        success = False
        reward_sum = 0.0
        steps = 0
        for action in replay_actions:
            obs, reward, done, info = env.step(action.astype(np.float32))
            reward_sum += float(reward)
            steps += 1
            try:
                success = bool(env.check_success())
            except Exception:
                success = bool(done and reward > 0)
            if success or done:
                break
        replay.update({"success": success, "steps": steps, "reward": reward_sum})
    env.close()

    np.save(output_dir / "gt_chunk.npy", gt_chunk)
    np.save(output_dir / "pred_hdf5_chunk.npy", pred_hdf5)
    np.save(output_dir / "pred_online_chunk.npy", pred_online)
    metrics = {
        "model_worker": hello,
        "instruction": instruction,
        "sample": {
            "timestep": int(original_timestep),
            "filtered_timestep": int(filtered_timestep),
            "gt_indices": [int(x) for x in gt_indices],
        },
        "frame_raw": frame_metrics(hdf5_frame_resized, online_frame),
        "frame_rot180": frame_metrics(hdf5_rot, online_rot),
        "pred_hdf5_vs_gt": action_metrics(pred_hdf5, gt_chunk),
        "pred_online_vs_gt": action_metrics(pred_online, gt_chunk),
        "pred_hdf5_vs_online": action_metrics(pred_hdf5, pred_online),
        "replay_gt_actions": replay,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
