# Draft — PyTorch issue (not yet filed)

Suggested title:

> [MPS] Conv2d with kernel == stride (ViT patch embedding, 32×32/32) is ~80× slower fwd+bwd than the equivalent reshape + matmul

Labels to request: `module: mps`, `module: performance`, `module: convolution`.

## Duplicate check (2026-09-05)

`gh issue list -R pytorch/pytorch --state all --search "<q> mps"` for `conv2d slow`, `conv2d
backward slow`, `patch embedding conv2d slow`, `conv2d stride kernel_size slow`, `conv2d large
kernel slow`, `conv2d slower than matmul`, `mps convolution backward performance`, `ViT slow
backward`, `CLIP slow`. **No existing report of this.** Related but distinct:

- [#192213](https://github.com/pytorch/pytorch/issues/192213) (closed) — `F.conv3d` at ~7 % of
  conv2d throughput on MPS: same genre (an MPS conv path far off the hardware), different op.
- [#192551](https://github.com/pytorch/pytorch/issues/192551) (closed) — compiled training 4×
  slower on MPS because layout optimisation forces channels_last: conv layout sensitivity, not
  this shape.
- Our own companion draft `01-mps-conv-backward-strided-grad-output.md` is about the *same conv*
  with a permuted+offset `grad_output` (200× slower again). This issue is the remaining gap once
  that grad is made contiguous.

---

## 🐛 Describe the bug

A `Conv2d` whose kernel equals its stride is a matmul over non-overlapping patches — the
standard ViT patch embedding. On MPS the convolution kernel is dramatically slower than that
matmul for this shape, forward and especially backward:

CLIP ViT-B/32's `Conv2d(3, 768, kernel_size=32, stride=32, bias=False)` on `[96, 3, 224, 224]`,
weights frozen (input gradient only), M1 Max, torch 2.14.0, means of 5 after 2 warm-ups,
`torch.mps.synchronize()` around the timed region:

| form | fp16 fwd | fp16 fwd + bwd(input) | fp32 fwd | fp32 fwd + bwd(input) |
|---|---|---|---|---|
| `Conv2d` (contiguous `grad_output`) | 20.5 ms | **488.2 ms** | 22.9 ms | **612.7 ms** |
| `reshape → [96, 49, 3072] @ W.T` | 3.1 ms | **6.3 ms** | 3.4 ms | **6.6 ms** |

That is ~7× on the forward and ~80–90× on forward + backward. The outputs agree to
`max|diff|` = 1e-3 in fp16 (one ulp at magnitude ~1) and 8e-6 in fp32. The conv's *backward*
is the pathological half: 468 ms for a gradient that the matmul form computes in ~3 ms.

The workload is 96 × 49 patches × (3072 → 768), i.e. ~22 GFLOP forward — about 5 ms at a
plausible MPS GEMM rate, which the matmul form hits. The conv path is running at well under
100 GFLOPS on this op.

Practical impact: this single op was **~47 % of an optimisation step** in
[big-sleep-mps](https://github.com/intrinsic-labs/big-sleep-mps) (CLIP-guided BigGAN, gradient
flows *through* CLIP into the image), and applies to anything that back-propagates through a
ViT patch embedding on Apple Silicon — CLIP guidance, feature inversion, adversarial examples,
textual inversion with a CLIP loss. Rewriting the patch embedding as a matmul took the step
from 1,002 ms to 520 ms. (The same conv also has a separate 200× slow path when the incoming
gradient is a permuted view with a storage offset — see the companion report on
`mps_convolution_backward` with a `cat()`-shaped `grad_output`; the numbers above are with that
already worked around, so they are the *fast* path.)

## Minimal repro

Standalone script, no models or downloads (also at
`scripts/mps_patch_embed_conv_repro.py` in the repo above):

```python
import sys, time, torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
C, D, P, H = 3, 768, 32, 224; G = H // P
dev = "mps"

def bench(label, fn, n=5, warm=2):
    for _ in range(warm): fn()
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize()
    print(f"  {label:44s} {(time.perf_counter() - t) / n * 1000:9.1f} ms")

for dtype in (torch.float16, torch.float32):
    conv = torch.nn.Conv2d(C, D, P, P, bias=False).to(dev, dtype).requires_grad_(False)
    W = conv.weight.reshape(D, -1)
    x = torch.randn(N, C, H, H, device=dev, dtype=dtype)
    def as_conv(xi):   return conv(xi).reshape(N, D, G * G).permute(0, 2, 1)
    def as_matmul(xi): return xi.reshape(N, C, G, P, G, P).permute(0, 2, 4, 1, 3, 5).reshape(N, G * G, C * P * P) @ W.t()
    with torch.no_grad():
        print(f"{dtype}: max|conv - matmul| = {(as_conv(x) - as_matmul(x)).abs().max().item():.2e}")
    for label, fn in (("Conv2d", as_conv), ("reshape + matmul", as_matmul)):
        bench(f"{label} forward (no_grad)", lambda: fn(x))
        def fwd_bwd(fn=fn):
            xi = x.detach().requires_grad_(); y = fn(xi); y.backward(torch.ones_like(y))
        bench(f"{label} forward + backward(input)", fwd_bwd)
```

Output on the machine below (pasted from `scripts/mps_patch_embed_conv_repro.py 96`):

```
torch 2.14.0  device=mps  batch=96  Conv2d(3, 768, kernel=32, stride=32, bias=False)
torch.float16: max|conv - matmul| = 1.95e-03
  Conv2d forward (no_grad)                          20.6 ms
  Conv2d forward + backward(input)                 504.2 ms
  reshape + matmul forward (no_grad)                 3.4 ms
  reshape + matmul forward + backward(input)         5.9 ms
torch.float32: max|conv - matmul| = 8.34e-06
  Conv2d forward (no_grad)                          23.3 ms
  Conv2d forward + backward(input)                 628.1 ms
  reshape + matmul forward (no_grad)                 3.7 ms
  reshape + matmul forward + backward(input)         6.7 ms
```

## Expected behavior

A kernel == stride convolution should run at (or be lowered to) the speed of the equivalent
GEMM — the forward is a single `[N·49, 3072] × [3072, 768]` matmul and the input-gradient is
its transpose. At minimum the backward should not be 75× the forward's matmul-equivalent.

## Versions

- PyTorch 2.14.0 (pip wheel), torchvision 0.29.0
- Python 3.12.13 (arm64)
- macOS 26.5.2 (build 25F84), Apple M1 Max, 32 GB
- `torch.backends.mps.is_available() == True`

(`python -m torch.utils.collect_env` output to be pasted at filing time.)
