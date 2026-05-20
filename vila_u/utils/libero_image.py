from __future__ import annotations

from typing import Any

import numpy as np
import torch


def rotate_libero_image_180(image: Any) -> Any:
    """Apply the LIBERO/OpenVLA 180-degree RGB preprocessing rotation."""

    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None and isinstance(image, Image.Image):
        transpose = getattr(Image, "Transpose", Image)
        return image.transpose(transpose.ROTATE_180)

    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            return image
        if image.ndim >= 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            return np.ascontiguousarray(np.flip(image, axis=(-2, -1)))
        return np.ascontiguousarray(np.flip(image, axis=(0, 1)))

    if isinstance(image, torch.Tensor):
        if image.ndim < 2:
            return image
        if image.ndim >= 3 and image.shape[0] in (1, 3, 4):
            return torch.flip(image, dims=(-2, -1))
        return torch.flip(image, dims=(-3, -2) if image.ndim >= 3 else (-2, -1))

    return image
