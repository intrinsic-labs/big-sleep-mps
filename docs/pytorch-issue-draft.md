# Draft — PyTorch issue (not yet filed)

Suggested title:

> [MPS] `mps_convolution_backward` ~200× slower when `grad_output` is a permuted view with a storage offset (e.g. the grad `torch.cat` produces for a ViT patch embedding)

Labels to request: `module: mps`, `module: performance`.

---

## 🐛 Describe the bug

On MPS, the backward of a plain `Conv2d` becomes ~200× slower when the incoming `grad_output` has one specific non-contiguous layout: a permuted view **with a non-zero storage offset and an over-allocated outer stride**. That is exactly the gradient autograd produces for a ViT patch-embedding conv, because the conv output is reshaped/permuted to `[N, L, C]` and a class token is prepended with `torch.cat(..., dim=1)` — the `cat` backward returns a `narrow()` of the incoming grad.

Contiguous grads, plain channels-last permutations, and the same permutation *without* the offset are all fast. Only the offset+permuted layout hits the slow path. Calling `.contiguous()` on the grad before it reaches the conv (a `register_hook` on the conv output) removes the slowdown; the results are numerically identical.

Concretely, for CLIP ViT-B/32's `Conv2d(3, 768, kernel_size=32, stride=32, bias=False)` on `[N, 3, 224, 224]`:

| batch | conv fwd+bwd, contiguous grad | via the `cat` graph (as in CLIP) | same + `.contiguous()` hook |
|---|---|---|---|
| 8 | 56.1 ms | **10,991.0 ms** | 60.3 ms |
| 96 | 610.3 ms | **131,068.2 ms** | 617.8 ms |

(Batch 96 from an earlier run of the same script; each slow case there takes ~2 minutes.)

Isolating the layout (batch 8, `conv(x).backward(g)` with a hand-built `g` of shape `[8, 768, 7, 7]`):

| `grad_output` layout | ms |
|---|---|
| contiguous | 56.1 |
| channels-last permutation `randn(N,7,7,768).permute(0,3,1,2)` | 46.6 |
| permuted, no offset `randn(N,49,768).permute(0,2,1).reshape(N,768,7,7)` | 46.3 |
| **permuted + offset** `randn(N,50,768)[:, 1:].permute(0,2,1).reshape(N,768,7,7)` — strides `(38400, 1, 5376, 768)`, storage_offset 768 | **11,322.6** |
| the same tensor after `.contiguous()` | 59.0 |

fp16 vs fp32 makes no difference (measured 10.90 s vs 10.86 s at batch 8 in the original investigation).

Practical impact: this is the whole cost of running anything that back-propagates *through* OpenAI CLIP's ViT on Apple Silicon (CLIP-guided optimisation, adversarial examples, feature inversion, textual inversion with a CLIP loss …). In [big-sleep-mps](https://github.com/intrinsic-labs/big-sleep-mps) one optimisation step went from ~128 s to ~1 s after adding the hook.

## Minimal repro

Standalone script (also at `scripts/mps_conv_backward_repro.py` in the repo above):

```python
import sys, time, torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
dev = "mps"
conv = torch.nn.Conv2d(3, 768, kernel_size=32, stride=32, bias=False).to(dev)
conv.requires_grad_(False)
x = torch.randn(N, 3, 224, 224, device=dev)

def timeit(label, fn, n=3):
    fn(); torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize()
    print(f"{label:64s} {(time.perf_counter() - t) / n * 1000:9.1f} ms")

def as_in_clip(make_contiguous):
    xi = x.detach().requires_grad_()
    y = conv(xi)                                  # [N, 768, 7, 7]
    if make_contiguous:
        y.register_hook(lambda g: g.contiguous())
    y = y.reshape(N, 768, -1).permute(0, 2, 1)    # [N, 49, 768]
    y = torch.cat([torch.zeros(N, 1, 768, device=dev), y], dim=1)  # [N, 50, 768]
    y.mean().backward()

def conv_backward(make_grad):
    xi = x.detach().requires_grad_()
    conv(xi).backward(make_grad())

cat_shaped = lambda: torch.randn(N, 50, 768, device=dev)[:, 1:].permute(0, 2, 1).reshape(N, 768, 7, 7)

g = cat_shaped()
print("strides", g.stride(), "offset", g.storage_offset())
timeit("CLIP patch-embed graph (conv -> permute -> cat)", lambda: as_in_clip(False))
timeit("same, grad .contiguous() hook on conv output", lambda: as_in_clip(True))
timeit("conv.backward(grad contiguous)", lambda: conv_backward(lambda: torch.randn(N, 768, 7, 7, device=dev)))
timeit("conv.backward(grad permuted, no offset)", lambda: conv_backward(lambda: torch.randn(N, 49, 768, device=dev).permute(0, 2, 1).reshape(N, 768, 7, 7)))
timeit("conv.backward(grad permuted + offset)  <-- SLOW", lambda: conv_backward(cat_shaped))
timeit("conv.backward(that grad .contiguous())", lambda: conv_backward(lambda: cat_shaped().contiguous()))
```

Output on the machine below:

```
torch 2.14.0 device=mps batch=8
cat-shaped grad: contiguous=False strides=(38400, 1, 5376, 768) offset=768
CLIP patch-embed graph (conv -> permute -> cat), fwd+bwd           10991.0 ms
same graph with grad .contiguous() hook on conv output                60.3 ms
conv.backward(grad = contiguous)                                      56.1 ms
conv.backward(grad = channels-last permutation)                       46.6 ms
conv.backward(grad = permuted, no offset)                             46.3 ms
conv.backward(grad = permuted + offset, as from cat)  <-- SLOW     11322.6 ms
conv.backward(grad = that .contiguous())                              59.0 ms
```

## Expected behavior

`mps_convolution_backward` should either handle the strided/offset `grad_output` at roughly the speed of the contiguous case, or make it contiguous itself (a `[8,768,7,7]` fp32 copy is ~1.2 MB; the observed penalty is ~10 s).

## Versions

- PyTorch 2.14.0 (pip wheel, `torch==2.14.0`), torchvision 0.29.0
- Python 3.12.13 (arm64)
- macOS 26.5.2 (build 25F84), Apple M1 Max, 32 GB
- `torch.backends.mps.is_available() == True`

(`python -m torch.utils.collect_env` output to be pasted at filing time.)
