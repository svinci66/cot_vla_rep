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
    parser.add_argument(
        "--demo-init-hdf5",
        default=None,
        help="Optional LIBERO HDF5 demo file whose init state should be used for reset.",
    )
    parser.add_argument(
        "--demo-name",
        default="demo_0",
        help="Demo group name used with --demo-init-hdf5.",
    )
    parser.add_argument(
        "--demo-names",
        nargs="+",
        default=None,
        help="Optional list of demo group names to evaluate with --demo-init-hdf5.",
    )
    parser.add_argument(
        "--demo-start",
        type=int,
        default=None,
        help="Optional first demo index for --demo-init-hdf5, e.g. 0 for demo_0.",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=None,
        help="Optional number of demos starting at --demo-start.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-dir", default=None, help="Optional directory for auto-named rollout summary JSON.")
    parser.add_argument("--save-rollout-video", action="store_true", help="Save per-episode rollout videos.")
    parser.add_argument("--video-dir", default=None, help="Directory for rollout mp4/gif files.")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-format", default="both", choices=("mp4", "gif", "both"))
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


def load_demo_init_state(hdf5_path: str, demo_name: str) -> np.ndarray:
    import h5py

    path = Path(hdf5_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Demo init HDF5 not found: {path}")

    with h5py.File(path, "r") as h5_file:
        data = h5_file["data"]
        if demo_name not in data:
            available = sorted(data.keys())[:10]
            raise KeyError(
                f"Demo {demo_name!r} not found in {path}. "
                f"Available examples: {available}"
            )
        demo = data[demo_name]
        if "init_state" in demo.attrs:
            return np.asarray(demo.attrs["init_state"])
        if "states" in demo and len(demo["states"]) > 0:
            return np.asarray(demo["states"][0])

    raise ValueError(f"No init_state attr or states[0] found for {demo_name!r} in {path}")


def resolve_demo_names(args: argparse.Namespace) -> list[str]:
    if args.demo_names:
        return list(args.demo_names)
    if args.demo_start is not None or args.demo_count is not None:
        start = 0 if args.demo_start is None else args.demo_start
        count = 1 if args.demo_count is None else args.demo_count
        if count <= 0:
            raise ValueError("--demo-count must be positive")
        return [f"demo_{idx}" for idx in range(start, start + count)]
    return [args.demo_name]


def obs_to_payload(obs: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float64:
                value = value.astype(np.float32)
            payload[key] = value
    return payload


def obs_frame(obs: dict[str, Any], camera: str) -> np.ndarray | None:
    frame = obs.get(camera)
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        frame = frame[..., :3]
    elif frame.ndim == 3 and frame.shape[0] >= 3:
        frame = np.moveaxis(frame[:3], 0, -1)
    else:
        return None
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def save_episode_video(
    frames: list[np.ndarray],
    video_dir: Path,
    episode_idx: int,
    fps: int,
    video_format: str,
    prefix: str = "episode",
) -> dict[str, str]:
    if not frames:
        return {}
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Saving rollout videos requires imageio. Install it in the LIBERO env: "
            "pip install imageio imageio-ffmpeg"
        ) from exc

    video_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if video_format in ("mp4", "both"):
        mp4_path = video_dir / f"{prefix}_{episode_idx:03d}.mp4"
        imageio.mimsave(mp4_path, frames, fps=fps)
        outputs["mp4"] = str(mp4_path)
    if video_format in ("gif", "both"):
        gif_path = video_dir / f"{prefix}_{episode_idx:03d}.gif"
        imageio.mimsave(gif_path, frames, fps=fps)
        outputs["gif"] = str(gif_path)
    return outputs


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
    init_states = None
    init_state_source = "benchmark"
    demo_names = []
    if args.demo_init_hdf5:
        demo_names = resolve_demo_names(args)
        init_state_source = "demo_hdf5"
    else:
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
    print(f"init_state_source: {init_state_source}")
    if args.demo_init_hdf5:
        print(f"demo_init_hdf5: {args.demo_init_hdf5}")
        print(f"demo_names: {demo_names}")
    if args.output_json is None and args.output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        args.output_json = str(
            output_dir / f"{args.suite}_task{args.task_id}_seed{args.seed}_{timestamp}_summary.json"
        )
    if args.output_json:
        print(f"output_json: {args.output_json}")
    video_dir = None
    if args.save_rollout_video:
        if args.video_dir:
            video_dir = Path(args.video_dir)
        elif args.output_dir:
            video_dir = Path(args.output_dir) / "videos"
        elif args.output_json:
            video_dir = Path(args.output_json).parent / "videos"
        else:
            video_dir = Path("outputs") / "libero_rollout_videos"
        print(f"video_dir: {video_dir}")
        print(f"video_format: {args.video_format}, video_fps: {args.video_fps}")

    # Lightweight readiness handshake.
    socket.send(pack({"type": "hello", "suite": args.suite, "task_id": args.task_id}))
    hello = unpack(socket.recv())
    print(f"model_worker: {hello}")

    results = []
    success_count = 0
    total_episodes = args.episodes * max(len(demo_names), 1)
    try:
        rollout_specs = []
        if args.demo_init_hdf5:
            for demo_name in demo_names:
                demo_init_state = load_demo_init_state(args.demo_init_hdf5, demo_name)
                for episode_idx in range(args.episodes):
                    rollout_specs.append((demo_name, demo_init_state, episode_idx))
        else:
            for episode_idx in range(args.episodes):
                rollout_specs.append((None, None, episode_idx))

        for rollout_idx, (demo_name, demo_init_state, episode_idx) in enumerate(rollout_specs):
            obs = env.reset()
            init_state_id = None
            if demo_init_state is not None:
                obs = env.set_init_state(demo_init_state)
            else:
                init_state_id = (args.init_state_offset + episode_idx) % len(init_states)
                obs = env.set_init_state(init_states[init_state_id])
            episode_reward = 0.0
            success = False
            final_info = {}
            start_time = time.time()
            frames = []
            if args.save_rollout_video:
                frame = obs_frame(obs, args.camera)
                if frame is not None:
                    frames.append(frame)

            for step_idx in range(args.max_steps):
                request = {
                    "type": "act",
                    "episode": rollout_idx,
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
                if args.save_rollout_video:
                    frame = obs_frame(obs, args.camera)
                    if frame is not None:
                        frames.append(frame)
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
                "rollout_index": rollout_idx,
                "init_state_source": init_state_source,
                "init_state_id": init_state_id,
                "demo_name": demo_name,
                "success": bool(success),
                "steps": step_idx + 1,
                "reward": episode_reward,
                "elapsed_sec": elapsed,
            }
            if args.save_failures or success:
                episode_result["final_info"] = final_info
            if args.save_rollout_video:
                episode_result["video_paths"] = save_episode_video(
                    frames=frames,
                    video_dir=video_dir,
                    episode_idx=episode_idx,
                    fps=args.video_fps,
                    video_format=args.video_format,
                    prefix=demo_name or "episode",
                )
            results.append(episode_result)
            demo_label = f" demo={demo_name}" if demo_name is not None else ""
            print(
                f"rollout={rollout_idx + 1}/{total_episodes}{demo_label} "
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
        "total_episodes": total_episodes,
        "success_count": success_count,
        "success_rate": success_count / max(total_episodes, 1),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "init_state_source": init_state_source,
        "demo_init_hdf5": args.demo_init_hdf5,
        "demo_name": args.demo_name if args.demo_init_hdf5 and not args.demo_names else None,
        "demo_names": demo_names if args.demo_init_hdf5 else None,
        "init_state_offset": args.init_state_offset,
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
