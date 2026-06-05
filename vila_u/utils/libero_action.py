from __future__ import annotations

import numpy as np


def libero_raw_actions_to_model_actions(actions):
    """Convert LIBERO raw actions to OpenVLA-style model-space actions.

    LIBERO raw gripper convention is -1=open, +1=close. OpenVLA-style
    normalized action space uses +1=open, -1=close, so only the gripper
    dimension changes sign. The first six motion dimensions are unchanged.
    """
    converted = np.asarray(actions, dtype=np.float32).copy()
    if converted.shape[-1] > 6:
        converted[..., 6] = -converted[..., 6]
    return np.clip(converted, -1.0, 1.0)


def model_actions_to_libero_raw_actions(actions):
    """Convert model-space actions back to LIBERO raw actions for env.step."""
    converted = np.asarray(actions, dtype=np.float32).copy()
    if converted.shape[-1] > 6:
        converted[..., 6] = -converted[..., 6]
    return np.clip(converted, -1.0, 1.0)
