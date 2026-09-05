"""Standalone repro: on MPS, a patch-embedding Conv2d (kernel == stride, e.g. 32x32/32)
is ~7x slower forward and ~80x slower fwd+bwd(input) than the identical reshape + matmul.

    python scripts/mps_patch_embed_conv_repro.py [batch=96]

No models or downloads. Random weights; the two forms are the same numbers to within one
fp16 ulp. Companion to docs/pytorch-issues/02-mps-patch-embed-conv2d-vs-matmul.md.
"""
import sys
import time

import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
C, D, P, H = 3, 768, 32, 224          # CLIP ViT-B/32: 3 -> 768, 32x32 patches on 224x224
G = H // P                            # 7x7 grid
dev = "mps"
assert torch.backends.mps.is_available(), "needs an MPS device"


def bench(label, fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    ms = (time.perf_counter() - t) / n * 1000
    print(f"  {label:44s} {ms:9.1f} ms")
    return ms


print(f"torch {torch.__version__}  device={dev}  batch={N}  Conv2d({C}, {D}, kernel={P}, stride={P}, bias=False)")
for dtype in (torch.float16, torch.float32):
    conv = torch.nn.Conv2d(C, D, P, P, bias=False).to(dev, dtype).requires_grad_(False)
    W = conv.weight.reshape(D, -1)                       # [768, 3072]
    x = torch.randn(N, C, H, H, device=dev, dtype=dtype)

    def as_conv(xi):                                      # -> [N, 49, 768]
        return conv(xi).reshape(N, D, G * G).permute(0, 2, 1)

    def as_matmul(xi):                                    # -> [N, 49, 768], same numbers
        p = xi.reshape(N, C, G, P, G, P).permute(0, 2, 4, 1, 3, 5).reshape(N, G * G, C * P * P)
        return p @ W.t()

    with torch.no_grad():
        diff = (as_conv(x) - as_matmul(x)).abs().max().item()
    print(f"{dtype}: max|conv - matmul| = {diff:.2e}")
    for label, fn in (("Conv2d", as_conv), ("reshape + matmul", as_matmul)):
        bench(f"{label} forward (no_grad)", lambda: fn(x))

        def fwd_bwd(fn=fn):
            xi = x.detach().requires_grad_()            # frozen weights: input grad only
            y = fn(xi)
            y.backward(torch.ones_like(y))
        bench(f"{label} forward + backward(input)", fwd_bwd)
