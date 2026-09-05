# Draft — PyTorch issue (not yet filed)

Suggested title:

> [MPS] `F.interpolate(mode="nearest")` backward sends gradient to the wrong source pixels for some sizes (e.g. 104→224, 468→224); forward is correct

Labels to request: `module: mps`, `module: correctness (silent)`, `module: autograd`.

## Duplicate check (2026-09-05)

`gh issue list -R pytorch/pytorch --state all --search "<q> mps"` for `interpolate nearest
backward wrong gradient`, `upsample_nearest2d backward incorrect`, `nearest backward gradient`,
`upsample nearest backward`, `interpolate gradient`, `F.interpolate grad`, `upsample_nearest2d_backward`,
`MPSGraph resize gradient`, `resizeNearestWithGradientTensor`. **No existing MPS report.**
Related but distinct:

- [#97135](https://github.com/pytorch/pytorch/issues/97135) (open, 2023) — "Incorrect gradient
  calculation for upsample nearest on **CUDA**": the same *symptom* on a different backend and
  a different kernel. Worth citing as precedent; the MPS backward is a separate implementation
  (MPSGraph `resizeNearestWithGradientTensor`), so this is a new issue rather than a duplicate.
- [#89277](https://github.com/pytorch/pytorch/issues/89277) (closed) — `upsample_nearest1d`
  not implemented on MPS: coverage, not correctness.

The reproducer below needs no models, weights or randomness, and the oracle is the forward's
own output (source indices encoded as pixel values, `bincount` of what the forward actually
produced) — so it does not depend on trusting any other backend's backward.

---

# MPS nearest-resize backward sends gradients to different source pixels

Found by GPT-6 Astra during the big-sleep-mps performance pass, 2026-09-05. **Not filed upstream.**

On an M1 Max, macOS 26.5.2 (25F84), torch 2.14.0, some `F.interpolate`
`mode="nearest"` shapes have identical CPU/MPS forward results but substantially
different input gradients. Big Sleep samples these sizes in its normal cutouts.
This is separate from the CLIP convolution layout issue.

## Reproduce without models or downloads

```sh
python scripts/mps_nearest_backward_repro.py
```

The script uses a contiguous, one-channel image whose values encode source pixel
indices. For `sum(resize(image))`, the exact derivative at each source pixel is
the number of times that pixel occurs in the output. Counting the **actual MPS
forward output's** indices supplies an independent derivative oracle; this does
not depend on trusting another backend's backward implementation.

Affected sizes observed include 104, 304, 348, 390, 454, and 468 resized to 224.
409 → 224 is a passing control. The script prints wrong-pixel counts and an
example mismatch for each shape.

| Resize | Wrong gradient pixels | Maximum error for an all-ones output gradient |
|---|---:|---:|
| 104 → 224 | 2,280 | 5 |
| 304 → 224 | 4,430 | 1 |
| 348 → 224 | 1,784 | 1 |
| 390 → 224 | 894 | 1 |
| 454 → 224 | 894 | 1 |
| 468 → 224 | 2,670 | 1 |
| 409 → 224 | 0 | 0 |

All seven forward outputs match CPU exactly. For example, at 104 → 224,
source pixel `(0, 12)` appears six times in the forward output, but MPS assigns
it gradient nine. At 468 → 224, pixel `(0, 116)` is read once but receives zero.

With random unit-scale upstream gradients and the actual sampled crop offsets,
maximum CPU/MPS input-gradient discrepancies ranged from 3.37 to 7.23. For a
96-cutout batch, explicitly gathering the forward source pixels agreed with the
CPU gradient to approximately 1e-5 (fp32 accumulation order), whereas the
resize-and-slice graph differed by approximately 4.94. Forward pixels agreed
exactly in these comparisons.

## Implementation boundary

The installed wheel reports source commit
`08187d9e0fba026dc8217405802ab5381dc88d90`. Its
[MPS resize implementation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/UpSample.mm)
uses a Metal upsample kernel for nearest forward, but delegates nearest 2D
backward to MPSGraph's `resizeNearestWithGradientTensor`, using floor rounding
and uncentered coordinates. The installed source was checked, rather than
inferring the implementation from CPU profiler names.

A coordinate-rounding mismatch at integral boundaries is a hypothesis, **not a
confirmed internal MPSGraph root cause**. The reproduction establishes the
forward/backward disagreement independently of that hypothesis. A backend fix
should pass the source-index/count oracle for all supported resize dimensions.

## Why the corrected kernel is opt-in in big-sleep-mps

`big_sleep/mps_cutouts.py` (a Metal kernel via `torch.mps.compile_shader`) implements batched
nearest sampling and scatters the gradient back to the exact forward source pixel.
`scripts/bench_cutouts.py` checks this against CPU autograd and measures both
fixed geometry and resampled geometry, including metadata construction/copies.

Correcting these gradients changes Big Sleep's optimization trajectory for the
same seed. The experiment also uses atomic fp32 accumulation, whose addition
order can vary, and supports only first derivatives of contiguous, single-image
fp32 RGB inputs. It is deliberately **off by default** (`BIG_SLEEP_METAL_CUTOUTS=1` /
`--metal_cutouts` turns it on): neither the bug fix nor a new cutout backend is silently
introduced as a performance-only change. See `docs/performance.md`.
