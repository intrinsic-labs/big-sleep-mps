"""CLIP's fixed normalization for generated tensor images."""
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _constants(device, dtype):
    # These fixed positive standard deviations need no per-call GPU validation.
    # Construct in the input dtype, as torchvision does, including float64 on CPU.
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device, dtype=dtype)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device, dtype=dtype)
    return mean.view(3, 1, 1), std.view(3, 1, 1)


def normalize_clip_image(image):
    """Normalize without mutating the input or reading a GPU scalar on the CPU.

    Subtract into a fresh tensor, then divide in place: the same arithmetic as
    torchvision Normalize, without its preliminary clone and fixed-std check.
    """
    mean, std = _constants(image.device, image.dtype)
    return image.sub(mean).div_(std)
