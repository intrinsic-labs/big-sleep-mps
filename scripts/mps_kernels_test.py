"""Correctness + timing of big_sleep.mps_kernels against the torch reference.
    PYTHONPATH=$PWD python scripts/mps_kernels_test.py
"""
import time, torch
from big_sleep.mps_kernels import quick_gelu, quick_gelu_reference

dev = "mps"
def sync(): torch.mps.synchronize()
def bench(label, fn, n=10, warm=3):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); print(f"{label:56s} {(time.perf_counter()-t)/n*1000:8.2f} ms")

for dtype in (torch.float16, torch.float32):
    x = torch.randn(50, 96, 3072, device=dev, dtype=dtype) * 3
    g = torch.randn_like(x)
    xr = x.clone().requires_grad_(); xk = x.clone().requires_grad_()
    yr = quick_gelu_reference(xr); yk = quick_gelu(xk)
    yr.backward(g); yk.backward(g)
    # oracle in fp32 on the same inputs
    x32 = x.float().detach().requires_grad_(); y32 = quick_gelu_reference(x32); y32.backward(g.float())
    print(f"{dtype}: fwd max|kernel-ref| {(yk-yr).abs().max().item():.2e} | fwd max|kernel-fp32oracle| {(yk.float()-y32).abs().max().item():.2e} vs |ref-fp32oracle| {(yr.float()-y32).abs().max().item():.2e}")
    print(f"{dtype}: bwd max|kernel-ref| {(xk.grad-xr.grad).abs().max().item():.2e} | bwd max|kernel-fp32oracle| {(xk.grad.float()-x32.grad).abs().max().item():.2e} vs |ref-fp32oracle| {(xr.grad.float()-x32.grad).abs().max().item():.2e}")
    def fb(fn):
        def run():
            xi = x.detach().requires_grad_(); fn(xi).backward(g)
        return run
    bench(f"{dtype} reference fwd+bwd [50,96,3072]", fb(quick_gelu_reference))
    bench(f"{dtype} metal kernel fwd+bwd [50,96,3072]", fb(quick_gelu))
    with torch.no_grad():
        bench(f"{dtype} reference fwd only", lambda: quick_gelu_reference(x))
        bench(f"{dtype} metal kernel fwd only", lambda: quick_gelu(x))
