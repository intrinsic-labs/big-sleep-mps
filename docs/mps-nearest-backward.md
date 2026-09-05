# MPS nearest-resize backward sends gradients to different source pixels

Found during the Astra performance pass, 2026-09-05. **Not filed upstream.**

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

## Why the performance experiment is not enabled

`scripts/cutout_metal.py` and its sibling Metal source implement batched nearest
sampling and scatter the gradient back to the exact forward source pixel.
`scripts/bench_cutouts.py` checks this against CPU autograd and measures both
fixed geometry and resampled geometry, including metadata construction/copies.

Correcting these gradients changes Big Sleep's optimization trajectory for the
same seed. The experiment also uses atomic fp32 accumulation, whose addition
order can vary, and supports only first derivatives of contiguous, single-image
fp32 RGB inputs. It is deliberately **not imported by the application**.
Neither the bug fix nor a new cutout backend is silently introduced as a
performance-only change.
