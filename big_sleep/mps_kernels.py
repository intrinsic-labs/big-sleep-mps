"""Hand-rolled Metal kernels for the ops that torch's MPS backend runs as several
unfused passes, compiled at import through `torch.mps.compile_shader` (no C++ extension,
no build step). Each has a pure-torch reference used on every other device and as the
correctness oracle in scripts/mps_kernels_test.py.

QuickGELU (CLIP's activation, x * sigmoid(1.702 x)) is the one elementwise op inside the
transformer blocks: 14.7M elements per block at Big Sleep's 96-cutout batch. Eager MPS
runs its forward as 3 kernels and its backward as ~6 (sigmoid, mul, sub, mul, mul, add),
each a full pass over the tensor; fused it is one pass each way.

LayerNorm (frozen weight/bias, input grad only): CLIP upcasts fp16 -> fp32 around every
LayerNorm for accuracy, which on MPS is two extra full copies forward and two backward per
LN, 26 LNs per image batch. A native fp16 F.layer_norm is 2x faster but 2x less accurate.
These kernels read/write fp16 and reduce in fp32: one threadgroup per row, simd-group
reductions, and they save (mean, rstd) per row so the backward is a single pass too.
"""
import torch
import torch.nn.functional as F

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

// ---- LayerNorm over the last dim, one threadgroup (LN_TG threads) per row ----
#define LN_TG 256

inline float tg_sum(float v, threadgroup float* scratch, uint lid, uint simd_lane, uint simd_id) {
    v = simd_sum(v);
    if (simd_lane == 0) scratch[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < LN_TG / 32; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

template <typename T>
kernel void layer_norm_fwd(constant T* x [[buffer(0)]],
                           constant float* w [[buffer(1)]],
                           constant float* b [[buffer(2)]],
                           device T* y [[buffer(3)]],
                           device float2* stats [[buffer(4)]],
                           constant uint& D [[buffer(5)]],
                           constant float& eps [[buffer(6)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[LN_TG / 32];
    constant T* xr = x + row * D;
    float s = 0.0f, ss = 0.0f;
    for (uint j = lid; j < D; j += LN_TG) { float v = float(xr[j]); s += v; ss += v * v; }
    s = tg_sum(s, scratch, lid, simd_lane, simd_id);
    ss = tg_sum(ss, scratch, lid, simd_lane, simd_id);
    float mean = s / float(D);
    float var = max(ss / float(D) - mean * mean, 0.0f);
    float rstd = rsqrt(var + eps);
    device T* yr = y + row * D;
    for (uint j = lid; j < D; j += LN_TG) yr[j] = T((float(xr[j]) - mean) * rstd * w[j] + b[j]);
    if (lid == 0) stats[row] = float2(mean, rstd);
}

// gx = rstd * (g*w - mean(g*w) - xhat * mean(g*w*xhat)), xhat = (x - mean) * rstd
template <typename T>
kernel void layer_norm_bwd(constant T* x [[buffer(0)]],
                           constant T* gy [[buffer(1)]],
                           constant float* w [[buffer(2)]],
                           constant float2* stats [[buffer(3)]],
                           device T* gx [[buffer(4)]],
                           constant uint& D [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[LN_TG / 32];
    constant T* xr = x + row * D;
    constant T* gr = gy + row * D;
    float mean = stats[row].x, rstd = stats[row].y;
    float s1 = 0.0f, s2 = 0.0f;
    for (uint j = lid; j < D; j += LN_TG) {
        float gw = float(gr[j]) * w[j];
        float xh = (float(xr[j]) - mean) * rstd;
        s1 += gw; s2 += gw * xh;
    }
    s1 = tg_sum(s1, scratch, lid, simd_lane, simd_id) / float(D);
    s2 = tg_sum(s2, scratch, lid, simd_lane, simd_id) / float(D);
    device T* out = gx + row * D;
    for (uint j = lid; j < D; j += LN_TG) {
        float gw = float(gr[j]) * w[j];
        float xh = (float(xr[j]) - mean) * rstd;
        out[j] = T(rstd * (gw - s1 - xh * s2));
    }
}

#define INSTANTIATE_LN(T, S)                                                                   \
template [[host_name("layer_norm_fwd_" #S)]] kernel void layer_norm_fwd<T>(                    \
    constant T*, constant float*, constant float*, device T*, device float2*, constant uint&,  \
    constant float&, uint, uint, uint, uint);                                                  \
template [[host_name("layer_norm_bwd_" #S)]] kernel void layer_norm_bwd<T>(                    \
    constant T*, constant T*, constant float*, constant float2*, device T*, constant uint&,     \
    uint, uint, uint, uint);
INSTANTIATE_LN(half, half)
INSTANTIATE_LN(float, float)
INSTANTIATE_LN(bfloat, bfloat)
"""

LN_THREADGROUP = 256

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


def layer_norm_reference(x, weight, bias, eps):
    """CLIP's LayerNorm: compute in fp32, return in the input dtype."""
    return F.layer_norm(x.float(), (x.shape[-1],), weight, bias, eps).to(x.dtype)


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x = x.contiguous()
        D = x.shape[-1]
        rows = x.numel() // D
        y = torch.empty_like(x)
        stats = torch.empty(rows, 2, device=x.device, dtype=torch.float32)
        getattr(_kernels(), f"layer_norm_fwd_{_SUFFIX[x.dtype]}")(
            x, weight, bias, y, stats, D, float(eps),
            threads=rows * LN_THREADGROUP, group_size=LN_THREADGROUP)
        ctx.save_for_backward(x, weight, stats)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, weight, stats = ctx.saved_tensors
        D = x.shape[-1]
        rows = x.numel() // D
        gy = gy.contiguous()
        gx = torch.empty_like(x)
        getattr(_kernels(), f"layer_norm_bwd_{_SUFFIX[x.dtype]}")(
            x, gy, weight, stats, gx, D,
            threads=rows * LN_THREADGROUP, group_size=LN_THREADGROUP)
        return gx, None, None, None


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim with fp32 statistics; fused Metal kernels on MPS for a
    frozen fp32 weight/bias (no weight grads), CLIP's upcast reference otherwise."""
    if (x.device.type == 'mps' and x.dtype in _SUFFIX and not weight.requires_grad
            and weight.dtype == torch.float32 and bias is not None and bias.dtype == torch.float32):
        return _LayerNorm.apply(x, weight.contiguous(), bias.contiguous(), eps)
    return layer_norm_reference(x, weight, bias, eps)
