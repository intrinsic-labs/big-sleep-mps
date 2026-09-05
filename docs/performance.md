---
status: live
verified: 2026-09-05 @fable
---

# Performance on Apple Silicon: 128 s → 0.45 s per step

How big-sleep-mps got from one optimisation step every two minutes to ~2 steps a second on an
M1 Max, what the profile said, what was hand-rolled in Metal, what did not help (with numbers),
why "same seed → same image" is not on the table, and where the floor is. This supersedes the two
per-agent notes from the 2026-09-05 pass, kept verbatim under [`history/`](history/); the brief
and the two agents' mailbox transcripts are under [`collaboration/`](collaboration/).

Machine for every number: Apple M1 Max 32 GB, macOS 26.5.2, torch 2.14.0, Python 3.12. Step
times are synchronised means from `scripts/bench_step.py --n 20` (20 timed steps after one
warm-up, seed 1). "512/96" is 512 px with 96 cutouts (the CLI defaults), "512/64" is `--fast`,
"128/8" is the small case where host overhead dominates.

## Headline

| config | original port | after PR #1 (hook + freeze) | **consolidated, default** | `--metal_cutouts` |
|---|---|---|---|---|
| 512 px, 96 cutouts (defaults) | 128,201 ms | 1,002 ms | **447 ms**¹ | 435 ms |
| 512 px, 64 cutouts (`--fast`) | 86,232 ms | 721 ms | **331 ms** | 327 ms |
| 128 px, 8 cutouts | 10,784 ms | 121 ms | **62 ms** | 63 ms |
| `dream "a pyramid made of ice" --fast`, wall incl. model load | — | 153.4 s | **75.8 s** | 74.7 s |

¹ Two runs of the same command, 447 and 454 ms — treat ±10 ms as noise at this size.

287× end to end at the defaults; 2.2× on top of PR #1. `BIG_SLEEP_REFERENCE_MATH=1` turns every
rounding-changing optimisation off and reproduces the PR #1 tree **bit for bit** (verified on the
consolidated tree: 30 seeded steps at 128/8, image / losses / latents / class logits all
`max|diff| = 0`), at 1,029 ms/step.

## 1. The profile

`scripts/profile_step.py 512 96` synchronises the device between stages, so these are
GPU-inclusive wall times (torch.profiler's per-op CPU times on MPS measure kernel *encoding*,
not execution, and were used only for op counts). On the PR #1 tree:

| stage | 512/96 | 128/8 |
|---|---|---|
| BigGAN forward | 81.9 ms | 22.8 |
| cutouts + cat + normalize | 9.6 | 1.9 |
| CLIP `encode_image` forward (batch 96) | 208.2 | 23.2 |
| losses (latent / class / similarity) | 4.8 | 2.3 |
| backward, all | 729.3 | 85.5 |
| — CLIP backward (isolated) | ~637 | |
| — cutouts backward (isolated) | 15.1 | 4.0 |
| — BigGAN backward (isolated) | 73.7 | 16.8 |
| Adam + EMA + zero_grad | 2.8 | 1.2 |
| **total** | **1,036.7** | **136.8** |

CLIP forward + backward was **82 %** of a default step, BigGAN 15 %, and the whole
cutout/loss/optimiser side 3 % (≈ 10 % at 128/8, where ~4,000 aten calls per step make the host
the bottleneck). Astra's actual-graph profiler (`scripts/profile_pipeline.py`, which instruments
the real random-crop graph) agreed: ~207/634 ms CLIP fwd/bwd, ~72/79 ms generator fwd/bwd.

The anomaly that pointed at the fix: with frozen weights CLIP's backward needs the same matmul
FLOPs as its forward (~845 GFLOP at batch 96) yet took 3× as long. `scripts/clip_microbench.py`
split the ViT — one `ResidualAttentionBlock` fwd+bwd is ~29 ms in fp16, so twelve blocks are
~350 ms, leaving ~480 ms of CLIP backward *outside* the blocks. That was `conv1`, the
32×32/stride-32 patch-embedding conv.

## 2. What changed, ranked (512/96 step, cumulative)

| # | change | step after | Δ | numerics vs previous |
|---|---|---|---|---|
| 0 | original port (`main` before PR #1) | 128,201 | | |
| 1 | **PR #1** — `.contiguous()` hook on CLIP `conv1`'s output grad (the `cat().narrow()` layout sends `mps_convolution_backward` down a ~11 s path) + freeze CLIP/BigGAN (no weight grads for ~200 M frozen params) | 1,002 | −127 s | bit-identical |
| 2 | **Patch embedding as reshape + matmul** (MPS only) — `clip.py` `VisualTransformer.patch_embed`. The conv is a matmul over non-overlapping patches; MPS's Conv2d kernel is pathological for this shape *even on the fast path*: 488 → 6.3 ms fwd+bwd at batch 96 | **520** | −482 ms | 1 fp16 ulp |
| 3 | **BigGAN bake** — spectral norm removed from 174 modules (it recomputed `weight_orig / sigma` every forward, eval included); `conv_to_rgb` sliced 128 → 3 out-channels (the generator computed 128 channels of a 3×3 conv at 512² and kept three, ~75 GFLOP/forward); conditional BN's five elementwise passes fused to one `addcmul` | **485** | −35 ms | SN bake bit-identical (when baked on the device — see §4); rgb slice 1.8e-7; BN 3e-4 on the [-1, 1] image |
| 4 | **Fused QuickGELU Metal kernels** (fwd + bwd, `mps_kernels.py`) — eager MPS ran the backward as ~6 passes; 1.83 → 0.57 ms per block | **461** | −24 ms | *closer* to fp32 than eager fp16 |
| 5 | **Fused LayerNorm Metal kernels** — fp16 in/out, fp32 statistics, one pass each way, `(mean, rstd)` saved for the backward; 0.76 → 0.20 ms per LN | **446** | −15 ms | identical error to CLIP's upcast reference |
| 6 | **Cached CLIP normalization** (`normalization.py`, Astra) — torchvision's `Normalize` re-made mean/std tensors and ran `(std == 0).any()` on the GPU every call (a hidden sync); 8.07 → 4.97 ms per batch of 96 | ~446 | −3 ms (in noise) | **bit-exact**, pixels and gradients; final PNGs byte-identical |
| — | **consolidated tree, measured** | **447–454** | | bit-identical to `perf/fable` HEAD (30 seeded steps) |
| 7 | **Metal cutout kernel**, opt-in (`mps_cutouts.py`, Astra) — all 96 crops sampled and resized in one launch, gradient scattered in one; 19.1 → 2.0 ms fwd+bwd | **435** | −12 to −19 ms | *corrects* an MPS gradient bug → different trajectory (§3) |

Rows 1–6 are on by default. Row 7 is `BIG_SLEEP_METAL_CUTOUTS=1` / `--metal_cutouts`.

Isolated numbers behind the rows: BigGAN-deep-512 batch 1 (`scripts/biggan_microbench.py 512`)
fwd 66.3 → 38.1 ms, fwd+bwd 139.0 → 106.6 ms, of which spectral-norm bake −5 ms, fused BN −24 ms
fwd (the 256²/512² blocks are memory-bound: five passes over a 134 MB activation became one),
`conv_to_rgb` slice 18.8 → 9.5 ms (`scripts/biggan_layers_bench.py`). QuickGELU `[50, 96, 3072]`
fp16 fwd+bwd 1.83 → 0.57 ms; LayerNorm `[50, 96, 768]` fp16 fwd+bwd 0.76 → 0.20 ms
(`scripts/mps_kernels_test.py`). Normalization batch 96 fwd+bwd 8.070 → 4.967 ms.

Profile after (`scripts/profile_step.py 512 96`, default path): BigGAN fwd 36 · cutouts 9 ·
CLIP fwd 178 · losses 6 · backward 234 (CLIP 152, cutouts 16, BigGAN 64) · Adam/EMA 1.

## 3. The Metal kernels

All three ship through `torch.mps.compile_shader` — no C++/ObjC++ extension, no build step, no
new dependency — wrapped in `autograd.Function`s with pure-torch references used on every other
device and as the correctness oracle.

- **QuickGELU** (`big_sleep/mps_kernels.py`): `x * sigmoid(1.702 x)` and its derivative, one
  pass each way, `half`/`float`/`bfloat` instantiations. Computes in fp32 internally, so it is
  strictly more accurate than eager fp16 (bwd error 1.95e-3 vs 8.8e-3 against an fp32 oracle).
- **LayerNorm** (same file): one 256-thread threadgroup per row, `simd_sum` + threadgroup scratch,
  fp16 in/out with fp32 statistics, saves `(mean, rstd)` so the backward is a single pass. Frozen
  weight/bias only (no weight grads). A native fp16 `F.layer_norm` is the same speed with twice
  the error — rejected.
- **Cutouts** (`big_sleep/mps_cutouts.py`, opt-in): forward takes three integers per crop (size,
  row, col) and writes the `[N, 3, 224, 224]` batch directly; backward scatters each output
  gradient to the source pixel the forward actually read, with fp32 atomic add via integer CAS
  (works on every Apple GPU generation). fp32, single square RGB image, first derivatives only;
  refuses to run under `torch.use_deterministic_algorithms(True)` because atomic summation order
  varies. The box sampler consumes the same random draws in the same order as the PyTorch loop,
  so for a given seed the cutout *geometry* is identical and only the resize arithmetic differs.
  Full-step gain is ~19 ms at 512/96 and nothing at 128/8, where the step is host-bound.

**Why the cutout kernel is off by default.** While building it Astra found that torch's MPS
`F.interpolate(mode="nearest")` **backward sends gradient to the wrong source pixels** for some
sizes Big Sleep samples every step (104, 304, 348, 390, 454, 468 → 224 among them; 409 → 224 is
clean): forward pixels match CPU exactly, but at 104 → 224, 2,280 gradient pixels are wrong (a
pixel read six times gets gradient nine). The Metal kernel scatters to the pixels the forward
actually read, agreeing with the CPU oracle to ~1e-5. That is a correctness *fix* — and it changes
the trajectory of every seeded run, so it is opt-in: images differ but are equivalent. Repro with
no models: `scripts/mps_nearest_backward_repro.py`; draft issue:
[`pytorch-issues/03-mps-nearest-backward-wrong-gradients.md`](pytorch-issues/03-mps-nearest-backward-wrong-gradients.md).

## 4. Parity: "same seed → same image" is not on the table, and why

Base-vs-base on MPS is bit-identical run to run (30 steps, `max|diff| = 0` on image, losses and
latents; two 200-step `--fast` runs gave byte-identical PNGs), so the harness is trustworthy — and
it shows that **any** change of rounding anywhere gives a different picture for the same seed.
Ablations at 128/8 × 30 steps against the PR #1 tree (`scripts/parity_dump.py` /
`parity_compare.py`):

| variant | step-0 loss | step-1 loss | final image PSNR vs base |
|---|---|---|---|
| all switches off / `BIG_SLEEP_REFERENCE_MATH=1` | identical | identical | ∞ (bit-identical) |
| spectral-norm bake computed on **CPU** (first version) | identical | −8.750 vs −8.702 | 8.7 dB |
| spectral-norm bake computed on **MPS** (shipped) | identical | identical | ∞ (bit-identical) |
| `conv_to_rgb` slice alone (1.8e-7 fwd diff) | identical | differs | 15.8 dB |
| everything on (consolidated default) | −9.570 vs −9.562 | −8.773 vs −8.702 | 12.8 dB |
| default vs `--metal_cutouts` | identical | identical | 8.3 dB |
| Astra's latent-moment vectorisation (1.1e-8 grad drift, withdrawn) | identical | identical | RGB MAE 0.303 after 200 steps |

A 1e-7 forward difference or one fp16 ulp is a different image after ~10 steps of Adam at
lr 0.07 through a 100×-scaled cosine loss: the optimisation is chaotic. The spectral-norm bake
is the worked example of how fine the line is — `weight_orig / sigma` computed on the CPU at
load time rounds differently from the same division on MPS every forward, and 30 steps later the
image was unrecognisable (8.7 dB); baked on the device it is bit-exact
(`scripts/sn_bake_grad_probe.py`), which is why `from_pretrained` moves the model to the device
before baking, and why the bake is on even in reference mode.

So the bar is **equivalent quality**, and the evidence is end to end: `dream "a pyramid made of
ice" --fast` (seed 0, 200 steps) gives `samples/mps_fast_a_pyramid_made_of_ice.png` on the PR #1
tree (153 s), `samples/perf_fast_a_pyramid_made_of_ice.png` on the default path here (76 s) and
`samples/metal_cutouts_fast_a_pyramid_made_of_ice.png` with the kernel on (75 s) — all three
unmistakably an ice pyramid, three compositions; final losses over 30 steps at 128/8 are −23.3
(base) vs −22.7 (default) vs −20.4 (`--metal_cutouts`), i.e. none is optimising worse in a way the
loss can see. Astra's withdrawn vectorisation (`history/perf-astra/rejected-moments.png` beside
`reference.png`) is the same lesson from the other lane: a 1 s wall saving was not worth silently
changing every seed's image.

**`BIG_SLEEP_REFERENCE_MATH=1`** is the escape hatch: one switch (`big_sleep/reference_math.py`)
that restores the conv patch embedding, eager QuickGELU/LayerNorm, unfused BN, the full
128-channel `conv_to_rgb` and PyTorch cutouts, reproducing pre-optimisation runs bit for bit at
~1,030 ms/step. The spectral-norm bake and the normalization cache stay on because they are exact.

## 5. What did not help (numbers)

- **dtype.** CLIP already runs in fp16 on MPS — `clip.load` leaves Linear/Conv/proj fp16 with fp32
  LayerNorms, and `encode_image` casts the batch. At batch 96, fwd / fwd+bwd: fp16-as-shipped
  189 / 839 ms; **fp32 201 / 979**; **bf16 272 / 1,254**. "Try fp16" was the baseline already.
- **BigGAN in fp16.** fwd+bwd 107 → 90 ms, but `max|diff| = 0.35` on the [-1, 1] image (17 % of
  full scale on some pixel). Too much drift for a default.
- **`torch.compile`** (inductor MPS backend, torch 2.14) on the ViT: 966 vs 979 ms fp32 fwd+bwd.
  It fuses a little pointwise work and nothing else on MPS today.
- **Attention rewrites.** Batch-first SDPA 15.3 ms, manual softmax attention 12.6 ms,
  `nn.MultiheadAttention` as shipped 13.2 ms per block fwd+bwd — no win; the matmuls dominate.
- **Nearest ×2 upsample alternatives** (BigGAN's 14 upsamples): `F.interpolate` 2.5 ms fwd+bwd at
  `[1, 128, 256, 256]`; `expand + reshape` 287 ms; `repeat_interleave` 127 ms. The 70 ms of CPU
  self-time the profiler showed on `upsample_nearest2d_backward` was queue back-pressure.
- **Native fp16 LayerNorm.** Same speed as the Metal kernel, 2× the error (6.7e-3 fwd).
- **Latent-moment vectorisation** (Astra): 2,750 → 532 aten calls, 3.12 → 0.71 ms locally, 1.1e-8
  gradient drift — and a different composition after 200 steps (RGB MAE 0.303). Withdrawn from the
  runtime; kept as `scripts/latent_moments_experiment.py`.
- **Batched indexed gather for cutouts** (Astra): fast with precomputed indices (10.3 → 3.6 ms) but
  19.2 vs 19.1 ms once index construction is included each step. The Metal kernel avoids that.
- **Adam `foreach=True`** 0.213 → 0.205 ms; **EMA** 0.090 ms; **startup** (imports 1.3 s, model init
  2.25 s, two HEAD requests 0.25 s) — nothing worth changing. **Allocator**: ~821 MB live tensors vs
  ~6 GB driver allocation is cached resources, no case for knobs.

## 6. Where the floor is

~450 ms at 512/96 is roughly CLIP 330 (fwd 178 + bwd 152), BigGAN 100, cutouts 25, everything
else 10.

- **CLIP is now matmul-bound.** Twelve blocks × 136 GFLOP fwd+bwd(input) = 1.63 TFLOP per step in
  ~330 ms ≈ 5 TFLOPS achieved. Big fp16 GEMMs through MPS top out around 8–9 TFLOPS on this GPU, so
  the matmul-only floor is ~190–200 ms; the ~130 ms above it is MPS's GEMM efficiency on
  `[4800, 768] × [768, 3072]`-class shapes and attention's many small `[1152, 50, 64]` bmms.
  Beating MPS's GEMM with a hand-written Metal kernel is a project, not an afternoon; expect
  ≤ 10–15 % from it.
- **BigGAN** (~100 ms) is memory-bound in the 256²/512² blocks. A fused BN-affine + ReLU kernel
  with the per-channel reductions its backward needs is the next candidate: maybe −10 to −15 ms.
  fp16 would take another ~17 ms but costs visible drift.
- **Cutouts** are done: the Metal kernel takes them from ~25 to ~5 ms when enabled.
- **Host overhead** dominates at 128/8 (62 ms for ~40 ms of GPU work): ~4,000 aten calls per
  step. There is no CUDA-graph equivalent for MPS; fewer, larger ops is the only lever, and the
  one candidate (moment vectorisation) was withdrawn for drift.

Realistic floor with everything above shipped: **~350–380 ms at 512/96** (2.7× the PR #1 tree),
with the last ~150 ms to the GEMM limit needing a better GEMM than MPS's.

## 7. Reproducing

```bash
source .venv/bin/activate; export PYTHONPATH=$PWD   # an editable install may point at another checkout
python scripts/bench_step.py 512 96 --n 20            # default path
BIG_SLEEP_METAL_CUTOUTS=1 python scripts/bench_step.py 512 96 --n 20
BIG_SLEEP_REFERENCE_MATH=1 python scripts/bench_step.py 512 96 --n 5   # the exact, slow path
python scripts/profile_step.py 512 96                 # stage breakdown; --ops adds the aten table
python scripts/profile_pipeline.py 512 96 --n 20      # actual-graph boundaries (random crops)
python scripts/clip_microbench.py 96 --compile        # dtype / block / attention / compile probes
python scripts/patch_embed_bench.py 96;  python scripts/mps_patch_embed_conv_repro.py 96
python scripts/biggan_microbench.py 512; python scripts/biggan_layers_bench.py 512
python scripts/mps_kernels_test.py                    # Metal kernels vs reference vs fp32 oracle
python scripts/bench_cutouts.py 512 96 --n 30         # torch loop vs gather vs Metal, with CPU oracle
python scripts/mps_nearest_backward_repro.py          # the resize-gradient bug, no models
python scripts/parity_dump.py 128 8 30 /tmp/a.pt [--no-patch-embed --no-gelu --no-ln --no-fused-bn --no-rgb-slice]
python scripts/parity_compare.py /tmp/a.pt /tmp/b.pt
BIG_SLEEP_DEVICE=cpu python -m pytest -q test         # 10 passed, 1 skipped (fp64 on MPS); Metal tests run when MPS is present
```

Gotchas met on the way: an editable install resolves `big_sleep` to wherever it was installed
from, not the checkout you are standing in (set `PYTHONPATH`; `bench_step.py` now pins its own
root); `copy.deepcopy` of an MPS module aliases parameter storage, so `.to(dtype)` on the copy
mutates the original — reload instead; a fresh `nn.Conv2d` built during model loading consumes
the CPU RNG before the seeded latents are drawn (the `conv_to_rgb` slice is done in place for
that reason); two processes timing on one GPU contaminate each other by ~2× — serialise them.

## 8. Upstream

Three PyTorch issues came out of this, drafted and not yet filed, under
[`pytorch-issues/`](pytorch-issues/): the `cat()`-shaped-grad conv backward slow path (200×),
the patch-embedding Conv2d vs matmul gap (80×), and the nearest-resize backward gradient bug.
Each draft opens with a duplicate search against `pytorch/pytorch`.

## Credits

The 2026-09-05 pass was two agents working in parallel on separate branches from one brief:
Claude **Fable 5.1** (`perf/fable`: profile, patch embedding, BigGAN bake, QuickGELU and
LayerNorm kernels, the parity study, the reference-math switch) and GPT-6 **Astra** via Codex
(`perf/astra`: normalization cache, the actual-graph profiler, the cutout kernel, the
nearest-backward bug and its oracle, the GPU lock, the withdrawn-for-drift vectorisation). PR #1
(the contiguous hook and the freeze) preceded it. Primary sources: [`collaboration/`](collaboration/).
