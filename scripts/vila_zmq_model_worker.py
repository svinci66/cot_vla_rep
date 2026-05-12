#!/usr/bin/env python
"""VILA-U model worker for LIBERO online rollout over ZMQ.

Run this in the VILA-U/model conda environment. It loads the checkpoint once,
receives observations from scripts/libero_zmq_env_server.py, and returns one
7-DoF action per request.
"""

from __future__ import annotations

import argparse
import traceback
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
    parser.add_argument("--action-index", type=int, default=0, help="Which action in the predicted chunk to execute.")
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

    _, model, image_processor, _ = load_pretrained_model(
        model_path=resolved_model_path,
        device=args.device,
    )
    model.eval()

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
            obs = request["obs"]
            instruction = request["instruction"]
            image = select_image(obs, args.camera)
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            with torch.no_grad():
                if args.subgoal_mode == "generated":
                    action_chunk = model.predict_action_with_generated_subgoal(
                        image=image,
                        instruction=instruction,
                        image_processor=image_processor,
                        cfg=args.cfg,
                    )
                else:
                    action_chunk = model.predict_action(
                        image=image,
                        instruction=instruction,
                        image_processor=image_processor,
                    )
            action_chunk = action_chunk.detach().cpu().float().numpy()
            action_index = min(max(args.action_index, 0), action_chunk.shape[0] - 1)
            action = np.clip(action_chunk[action_index], -1.0, 1.0).astype(np.float32)
            socket.send(pack({"action": action, "action_index": action_index}))
        except Exception as exc:
            socket.send(pack({"error": str(exc), "traceback": traceback.format_exc()}))

    socket.close(0)


if __name__ == "__main__":
    main()
