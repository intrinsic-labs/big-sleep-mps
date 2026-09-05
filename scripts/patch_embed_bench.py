"""CLIP's patch-embedding conv (32x32, stride 32) vs the same op as reshape + matmul, on MPS.

    PYTHONPATH=$PWD python scripts/patch_embed_bench.py 96
"""
import sys, time, torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
dev = "mps"


def sync(): torch.mps.synchronize()


def bench(label, fn, n=5, warm=2):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); ms = (time.perf_counter() - t) / n * 1000
    print(f"{label:58s} {ms:9.1f} ms"); return ms


for dtype in (torch.float16, torch.float32):
    conv = torch.nn.Conv2d(3, 768, 32, 32, bias=False).to(dev, dtype).requires_grad_(False)
    W = conv.weight.reshape(768, -1)  # [768, 3*32*32]
    x = torch.randn(N, 3, 224, 224, device=dev, dtype=dtype)

    def patchify_matmul(xi):  # -> [N, 49, 768], i.e. already the [N, L, D] layout CLIP wants
        p = xi.reshape(N, 3, 7, 32, 7, 32).permute(0, 2, 4, 1, 3, 5).reshape(N, 49, 3 * 32 * 32)
        return p @ W.t()

    def conv_path(xi):
        return conv(xi).reshape(N, 768, 49).permute(0, 2, 1)

    with torch.no_grad():
        print(f"{dtype}: max|conv - matmul| =", (conv_path(x) - patchify_matmul(x)).abs().max().item())

    for label, fn in (("conv2d", conv_path), ("reshape+matmul", patchify_matmul)):
        bench(f"{dtype} {label} fwd (no_grad)", lambda: fn(x))
        def fb():
            xi = x.detach().requires_grad_()
            y = fn(xi); y.backward(torch.ones_like(y))
        bench(f"{dtype} {label} fwd+bwd(input)", fb)
