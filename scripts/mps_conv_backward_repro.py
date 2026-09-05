"""Minimal repro: mps_convolution_backward is ~200x slower for one specific
non-contiguous grad_output layout.

Shape is CLIP ViT-B/32's patch embedding: Conv2d(3, 768, k=32, s=32, bias=False)
on [N, 3, 224, 224] -> [N, 768, 7, 7]. In CLIP the conv output is
reshaped/permuted to [N, 49, 768] and a class token is prepended with
`torch.cat(..., dim=1)`. Autograd therefore hands the conv a grad_output that is
a permuted *and offset* view of a [N, 50, 768] buffer:

    shape (N, 768, 7, 7)  strides (50*768, 1, 7*768, 768)  storage_offset 768

A contiguous grad, a plain channels-last permutation, or the same permutation
without the offset/over-allocated batch stride are all fast. Only the
offset+permuted layout hits the slow path. Making the grad contiguous before
it reaches the conv (a `register_hook(lambda g: g.contiguous())` on the conv
output) removes the slowdown entirely; results are numerically identical.

Run:  python scripts/mps_conv_backward_repro.py [batch]
"""
import sys
import time

import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
dev = "mps" if torch.backends.mps.is_available() else "cpu"
sync = torch.mps.synchronize if dev == "mps" else (lambda: None)

conv = torch.nn.Conv2d(3, 768, kernel_size=32, stride=32, bias=False).to(dev)
conv.requires_grad_(False)
x = torch.randn(N, 3, 224, 224, device=dev)


def timeit(label, fn, n=3):
    fn(); sync()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    print(f"{label:64s} {(time.perf_counter() - t) / n * 1000:9.1f} ms", flush=True)


def as_in_clip(make_contiguous):
    xi = x.detach().requires_grad_()
    y = conv(xi)                                   # [N, 768, 7, 7]
    if make_contiguous:
        y.register_hook(lambda g: g.contiguous())  # the one-line fix
    y = y.reshape(N, 768, -1).permute(0, 2, 1)     # [N, 49, 768]
    cls = torch.zeros(N, 1, 768, device=dev)
    y = torch.cat([cls, y], dim=1)                 # [N, 50, 768]  <- the trigger
    y.mean().backward()


def conv_backward(make_grad):
    """conv.backward() with a hand-built grad_output of a given layout."""
    xi = x.detach().requires_grad_()
    conv(xi).backward(make_grad())


def cat_shaped():   # what autograd produces from the cat above
    return torch.randn(N, 50, 768, device=dev)[:, 1:].permute(0, 2, 1).reshape(N, 768, 7, 7)


def permuted_no_offset():
    return torch.randn(N, 49, 768, device=dev).permute(0, 2, 1).reshape(N, 768, 7, 7)


def channels_last():
    return torch.randn(N, 7, 7, 768, device=dev).permute(0, 3, 1, 2)


print(f"torch {torch.__version__} device={dev} batch={N}")
g = cat_shaped()
print(f"cat-shaped grad: contiguous={g.is_contiguous()} strides={g.stride()} offset={g.storage_offset()}")
timeit("CLIP patch-embed graph (conv -> permute -> cat), fwd+bwd", lambda: as_in_clip(False))
timeit("same graph with grad .contiguous() hook on conv output", lambda: as_in_clip(True))
timeit("conv.backward(grad = contiguous)", lambda: conv_backward(lambda: torch.randn(N, 768, 7, 7, device=dev)))
timeit("conv.backward(grad = channels-last permutation)", lambda: conv_backward(channels_last))
timeit("conv.backward(grad = permuted, no offset)", lambda: conv_backward(permuted_no_offset))
timeit("conv.backward(grad = permuted + offset, as from cat)  <-- SLOW", lambda: conv_backward(cat_shaped))
timeit("conv.backward(grad = that .contiguous())", lambda: conv_backward(lambda: cat_shaped().contiguous()))
