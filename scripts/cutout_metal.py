"""Experimental fp32 Metal cutouts; deliberately not enabled in BigSleep.

The backward scatters to the exact pixels read in the forward. It therefore
differs from affected MPSGraph nearest-resize gradients (see the repro script).
Atomic accumulation can also change fp32 summation order. First derivatives only.
"""
from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(Path(__file__).with_suffix(".metal").read_text())


class _Cutouts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, boxes):
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("Experimental Metal cutouts use nondeterministic atomic accumulation")
        assert image.device.type == "mps" and image.dtype == torch.float32
        assert image.is_contiguous() and image.shape[:2] == (1, 3)
        assert image.shape[2] == image.shape[3]
        assert boxes.device == image.device and boxes.dtype == torch.int32
        assert boxes.is_contiguous() and boxes.ndim == 2 and boxes.shape[1] == 3
        ctx.save_for_backward(boxes)
        ctx.width = image.shape[-1]
        out = image.new_empty((len(boxes), 3, 224, 224))
        _library().cutouts_forward(image, boxes, out, ctx.width,
                                   threads=out.numel(), group_size=256)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        boxes, = ctx.saved_tensors
        result = grad.new_zeros((1, 3, ctx.width, ctx.width))
        _library().cutouts_backward(grad.contiguous(), boxes, result, ctx.width,
                                    threads=grad.numel(), group_size=256)
        return result, None


metal_cutouts = _Cutouts.apply
