"""Hand-rolled Metal kernels for the ops that torch's MPS backend runs as several
unfused passes, compiled at import through `torch.mps.compile_shader` (no C++ extension,
no build step). Each has a pure-torch reference used on every other device and as the
correctness oracle in scripts/mps_kernels_test.py.

QuickGELU (CLIP's activation, x * sigmoid(1.702 x)) is the one elementwise op inside the
transformer blocks: 14.7M elements per block at Big Sleep's 96-cutout batch. Eager MPS
runs its forward as 3 kernels and its backward as ~6 (sigmoid, mul, sub, mul, mul, add),
each a full pass over the tensor; fused it is one pass each way.
"""
import torch

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void quick_gelu_fwd(constant T* x [[buffer(0)]],
                           device T* y [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    float v = float(x[i]);
    y[i] = T(v / (1.0f + exp(-1.702f * v)));
}

template <typename T>
kernel void quick_gelu_bwd(constant T* x [[buffer(0)]],
                           constant T* gy [[buffer(1)]],
                           device T* gx [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    float v = float(x[i]);
    float s = 1.0f / (1.0f + exp(-1.702f * v));
    gx[i] = T(float(gy[i]) * s * (1.0f + 1.702f * v * (1.0f - s)));
}

#define INSTANTIATE(T, S)                                                            \
template [[host_name("quick_gelu_fwd_" #S)]] kernel void quick_gelu_fwd<T>(         \
    constant T*, device T*, uint);                                                   \
template [[host_name("quick_gelu_bwd_" #S)]] kernel void quick_gelu_bwd<T>(         \
    constant T*, constant T*, device T*, uint);
INSTANTIATE(half, half)
INSTANTIATE(float, float)
INSTANTIATE(bfloat, bfloat)
"""

_SUFFIX = {torch.float16: "half", torch.float32: "float", torch.bfloat16: "bfloat"}
_lib = None


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


def quick_gelu_reference(x):
    return x * torch.sigmoid(1.702 * x)


class _QuickGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        getattr(_kernels(), f"quick_gelu_fwd_{_SUFFIX[x.dtype]}")(x, y)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        gy = gy.contiguous()
        gx = torch.empty_like(x)
        getattr(_kernels(), f"quick_gelu_bwd_{_SUFFIX[x.dtype]}")(x, gy, gx)
        return gx


def quick_gelu(x):
    """x * sigmoid(1.702 x); fused Metal kernels on MPS, reference math elsewhere."""
    if x.device.type == 'mps' and x.dtype in _SUFFIX:
        return _QuickGELU.apply(x)
    return quick_gelu_reference(x)
