#!/usr/bin/env python
"""VILA-U model worker for LIBERO online rollout over ZMQ.

Run this in the VILA-U/model conda environment. It loads the checkpoint once,
receives observations from scripts/libero_zmq_env_server.py, and returns one
7-DoF action per request.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
import torch
import zmq

from vila_u.model.builder import load_pretrained_model
from vila_u.train.utils import get_checkpoint_path

m.patch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VILA-U ZMQ model worker for LIBERO online eval.")
    parser.add_argument("--model-path", required=True, help="Checkpoint or output directory.")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera", default="agentview_image")
    parser.add_argument("--subgoal-mode", choices=("none", "generated"), default="none")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG for generated subgoal mode.")
    parser.add_argument("--debug-jsonl", default=None, help="Optional path for per-chunk action token/bin debug records.")
    return parser.parse_args()


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_dir():
        checkpoint_path, _ = get_checkpoint_path(str(path))
        if checkpoint_path is not None:
            return checkpoint_path
        return str(path)

    if not path.exists() and path.name.startswith("tmp-checkpoint-"):
        alt = path.with_name(path.name.replace("tmp-checkpoint-", "checkpoint-", 1))
        if alt.is_dir():
            return str(alt)

    raise FileNotFoundError(f"Model path does not exist: {path}")


def pack(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, default=m.encode, use_bin_type=True)


def unpack(payload: bytes) -> dict[str, Any]:
    return msgpack.unpackb(payload, object_hook=m.decode, raw=False)


def select_image(obs: dict[str, Any], camera: str) -> np.ndarray:
    if camera in obs:
        return obs[camera]
    fallback_keys = ["agentview_image", "agentview_rgb", "agentview_image_rgb"]
    for key in fallback_keys:
        if key in obs:
            return obs[key]
    image_keys = [key for key, value in obs.items() if isinstance(value, np.ndarray) and value.ndim == 3]
    if image_keys:
        return obs[image_keys[0]]
    raise KeyError(f"No image key {camera!r}; available keys: {sorted(obs.keys())}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    resolved_model_path = resolve_model_path(args.model_path)
    print("=" * 80)
    print("VILA-U ZMQ Model Worker")
    print("=" * 80)
    print(f"bind: {args.bind}")
    print(f"model: {resolved_model_path}")
    print(f"device: {args.device}")
    print(f"subgoal_mode: {args.subgoal_mode}")
    if args.debug_jsonl:
        print(f"debug_jsonl: {args.debug_jsonl}")

    _, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        device=args.device,
    )
    model.eval()

    action_queue = deque()
    current_episode = None
    chunk_id = 0
    debug_file = None
    if args.debug_jsonl:
        debug_path = Path(args.debug_jsonl)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path.open("a", encoding="utf-8")

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)
    print("ready")

    while True:
        request = unpack(socket.recv())
        request_type = request.get("type", "act")
        if request_type == "hello":
            socket.send(pack({"status": "ok", "model_path": resolved_model_path, "device": args.device}))
            continue
        if request_type == "close":
            socket.send(pack({"status": "closed"}))
            break

        try:
            episode = request.get("episode")
            step = request.get("step")
            if episode != current_episode:
                action_queue.clear()
                current_episode = episode

            obs = request["obs"]
            instruction = request["instruction"]
            image = select_image(obs, args.camera)
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            if not action_queue:
                with torch.no_grad():
                    if args.subgoal_mode == "generated":
                        prediction = model.predict_action_with_generated_subgoal(
                            image=image,
                            instruction=instruction,
                            image_processor=image_processor,
                            cfg=args.cfg,
                            return_debug=True,
                        )
                    else:
                        prediction = model.predict_action(
                            image=image,
                            instruction=instruction,
                            image_processor=image_processor,
                            return_debug=True,
                        )
                if isinstance(prediction, dict):
                    action_chunk_tensor = prediction["actions"]
                else:
                    action_chunk_tensor = prediction
                    prediction = {"actions": action_chunk_tensor}
                action_chunk = action_chunk_tensor.detach().cpu().float().numpy()
                cur_chunk_id = chunk_id
                chunk_id += 1
                for action_index, chunk_action in enumerate(action_chunk):
                    action_queue.append((cur_chunk_id, action_index, chunk_action))

                if debug_file is not None:
                    debug_record = {
                        "episode": episode,
                        "step": step,
                        "chunk_id": cur_chunk_id,
                        "subgoal_mode": args.subgoal_mode,
                        "action_chunk": to_jsonable(action_chunk),
                    }
                    for key in ("predicted_bins", "action_token_ids", "generated_subgoal_codes"):
                        if key in prediction:
                            debug_record[key] = to_jsonable(prediction[key])
                    debug_file.write(json.dumps(debug_record) + "\n")
                    debug_file.flush()

            cur_chunk_id, action_index, action = action_queue.popleft()
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            socket.send(pack({
                "action": action,
                "action_index": action_index,
                "chunk_id": cur_chunk_id,
                "queue_remaining": len(action_queue),
            }))
        except Exception as exc:
            socket.send(pack({"error": str(exc), "traceback": traceback.format_exc()}))

    if debug_file is not None:
        debug_file.close()
    socket.close(0)


if __name__ == "__main__":
    main()
