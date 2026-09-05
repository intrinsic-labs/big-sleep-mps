"""Batched nearest-neighbour cutouts as one Metal kernel each way (MPS only, opt-in).

Big Sleep's forward takes `num_cutouts` random square crops of the generated image and
resizes each to 224x224 with `F.interpolate(mode="nearest")`; the default step does that
96 times, each a slice + resize + (in backward) a resize-backward + slice-backward, and
the random sizes defeat MPS's per-shape graph cache. This kernel takes all crops in one
launch from three integers per crop (size, row offset, column offset) and scatters the
gradient back to exactly the pixels the forward read: 19.1 -> 2.0 ms fwd+bwd at 512/96.

It is OFF by default (`BIG_SLEEP_METAL_CUTOUTS=1`, `--metal_cutouts`, or
`Imagine(metal_cutouts=True)`) because it CHANGES GRADIENTS: torch's MPS nearest-resize
backward sends gradient to the wrong source pixels for some crop sizes (see
docs/pytorch-issues/03-mps-nearest-backward-wrong-gradients.md), so the corrected scatter
here produces a different -- but equivalent -- image for the same seed. Forward pixels
are identical; fp32 atomic accumulation in the backward also makes its summation order
non-deterministic, so it refuses to run under torch.use_deterministic_algorithms(True).
First derivatives only. Written by GPT-6 Astra during the 2026-09-05 performance pass.
"""
import os

import torch
from torch.autograd.function import once_differentiable

METAL_CUTOUTS_DEFAULT = os.environ.get('BIG_SLEEP_METAL_CUTOUTS', '') not in ('', '0')

OUT = 224  # CLIP's input resolution

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// Boxes are (size, row offset, column offset). No full pixel-index buffer.
inline uint source_index(uint index, device const int* boxes, uint width) {
    uint col = index % 224;
    uint row = (index / 224) % 224;
    uint channel = (index / (224 * 224)) % 3;
    uint crop = index / (3 * 224 * 224);
    float scale = float(boxes[crop * 3]) / 224.0f;
    uint y = uint(float(row) * scale) + boxes[crop * 3 + 1];
    uint x = uint(float(col) * scale) + boxes[crop * 3 + 2];
    return (channel * width + y) * width + x;
}

kernel void cutouts_forward(device const float* image,
                            device const int* boxes,
                            device float* out,
                            constant long& width,
                            uint index [[thread_position_in_grid]]) {
    out[index] = image[source_index(index, boxes, uint(width))];
}

kernel void cutouts_backward(device const float* grad,
                             device const int* boxes,
                             device atomic_uint* result,
                             constant long& width,
                             uint index [[thread_position_in_grid]]) {
    uint destination = source_index(index, boxes, uint(width));
    // fp32 atomic add via integer CAS also works on earlier Apple GPUs.
    uint old = atomic_load_explicit(result + destination, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        result + destination, &old,
        as_type<uint>(as_type<float>(old) + grad[index]),
        memory_order_relaxed, memory_order_relaxed)) {}
}
"""

_lib = None


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


def available():
    return torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader')


class _Cutouts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, boxes):
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("Metal cutouts accumulate their backward with fp32 atomics "
                               "(non-deterministic order); run without --metal_cutouts under "
                               "torch_deterministic")
        if image.device.type != 'mps' or image.dtype != torch.float32:
            raise TypeError("Metal cutouts need an fp32 MPS image")
        if image.ndim != 4 or image.shape[:2] != (1, 3) or image.shape[2] != image.shape[3]:
            raise ValueError(f"Metal cutouts expect one square RGB image [1, 3, W, W], got {tuple(image.shape)}")
        if boxes.device != image.device or boxes.dtype != torch.int32 or boxes.ndim != 2 or boxes.shape[1] != 3:
            raise ValueError("boxes must be an int32 [num_cutouts, 3] tensor of (size, row, col) on the image device")
        image = image.contiguous()
        boxes = boxes.contiguous()
        ctx.save_for_backward(boxes)
        ctx.width = image.shape[-1]
        out = image.new_empty((len(boxes), 3, OUT, OUT))
        _kernels().cutouts_forward(image, boxes, out, ctx.width, threads=out.numel(), group_size=256)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        boxes, = ctx.saved_tensors
        result = grad.new_zeros((1, 3, ctx.width, ctx.width))
        _kernels().cutouts_backward(grad.contiguous(), boxes, result, ctx.width,
                                    threads=grad.numel(), group_size=256)
        return result, None


def metal_cutouts(image, boxes):
    """All `len(boxes)` nearest-resized crops of `image` ([1, 3, W, W] fp32 on MPS) as one
    [N, 3, 224, 224] batch; `boxes` is int32 [N, 3] of (size, row_offset, col_offset)."""
    return _Cutouts.apply(image, boxes)
