"""Does CLIP's fp32-upcast LayerNorm (fp16 x -> fp32 -> LN -> fp16) cost anything on MPS
versus a native fp16 LayerNorm, and how far apart are they numerically?
    PYTHONPATH=$PWD python scripts/layernorm_cast_bench.py
"""
import time, torch, torch.nn.functional as F
dev = "mps"
def sync(): torch.mps.synchronize()
def bench(label, fn, n=10, warm=3):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); print(f"{label:56s} {(time.perf_counter()-t)/n*1000:8.2f} ms")

L, N, D = 50, 96, 768
x = (torch.randn(L, N, D, device=dev) * 2).half()
w = torch.randn(D, device=dev); b = torch.randn(D, device=dev)
g = torch.randn_like(x)
def ln_cast(t): return F.layer_norm(t.float(), (D,), w, b, 1e-5).half()
def ln_half(t): return F.layer_norm(t, (D,), w.half(), b.half(), 1e-5)
ref = F.layer_norm(x.float(), (D,), w, b, 1e-5)
with torch.no_grad():
    print(f"fwd max|cast - fp32 ref| {(ln_cast(x).float()-ref).abs().max().item():.2e}   max|half - fp32 ref| {(ln_half(x).float()-ref).abs().max().item():.2e}")
xr = x.float().requires_grad_(); F.layer_norm(xr, (D,), w, b, 1e-5).backward(g.float()); gref = xr.grad
for name, fn in (("cast", ln_cast), ("half", ln_half)):
    xi = x.clone().requires_grad_(); fn(xi).backward(g)
    print(f"bwd max|{name} - fp32 ref| {(xi.grad.float()-gref).abs().max().item():.2e}")
    def fb():
        xi = x.detach().requires_grad_(); fn(xi).backward(g)
    bench(f"LN {name} fwd+bwd [50,96,768]", fb)
    with torch.no_grad(): bench(f"LN {name} fwd", lambda: fn(x))
