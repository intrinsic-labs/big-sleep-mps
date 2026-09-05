---
status: historical
verified: 2026-09-05 @fable
---

> **Historical.** Fable 5.1's notes from the 2026-09-05 performance pass, as written on branch
> `perf/fable`, kept verbatim except for path fixes. The consolidated, current account is
> [`docs/performance.md`](../performance.md); the PyTorch issue drafts referenced here now live
> under [`docs/pytorch-issues/`](../pytorch-issues/). The `samples/perf_fable_fast_…` image referenced in §4
> was rendered before the branch's last commit; it is replaced by `samples/perf_fast_a_pyramid_made_of_ice.png`,
> rendered from the consolidated tree, which is bit-identical to this branch's final HEAD.


# big-sleep-mps performance pass — Fable's notes (branch `perf/fable`)

M1 Max 32 GB, macOS 26.5, torch 2.14.0, Python 3.12. Every number below is a synchronised
mean from `scripts/bench_step.py` (≥ 10 steps after warm-up, `--n 20` for the final row) or
from one of the microbenchmarks under `scripts/`, run under the shared GPU lock so nothing
else was on the GPU. Base is `fix/mps-contiguous-grad` (1,002 / 721 / 121 ms at 512/96,
512/64, 128/8 — Astra's `--n 20` baseline, cited rather than re-measured). Lane split with
GPT-6 Astra: I took CLIP fwd/bwd and BigGAN; Astra took cutouts, losses, optimiser/EMA,
syncs and pipeline structure (`docs/history/perf-notes-astra.md`).

## Headline

| config | base | `perf/fable` | speed-up |
|---|---|---|---|
| 512 px, 96 cutouts (CLI defaults) | 1,002 ms | **446 ms** | 2.25× |
| 512 px, 64 cutouts (`--fast`) | 721 ms | **332 ms** | 2.17× |
| 128 px, 8 cutouts | 121 ms | **64 ms** | 1.9× |
| `dream "a pyramid made of ice" --fast`, wall incl. model load | 153.4 s | **76.2 s** | 2.0× |

Everything is on by default. `BIG_SLEEP_REFERENCE_MATH=1` turns every change on this
branch except the (exact) spectral-norm bake back to the upstream form and reproduces the base branch **bit for bit** (verified:
30 seeded steps at 128/8, image / losses / latents all `max|diff| = 0`, at 999 ms/step).

## 1. Where the 1.02 s actually went

`scripts/profile_step.py 512 96` (device synchronised between stages, so these are
GPU-inclusive wall times; torch.profiler's per-op CPU times on MPS only measure kernel
*encoding* and were used for op counts, not time):

| stage | 512/96 | 128/8 |
|---|---|---|
| BigGAN forward | 81.9 ms | 22.8 |
| cutouts + cat + normalize | 9.6 | 1.9 |
| CLIP `encode_image` forward (batch 96) | 208.2 | 23.2 |
| losses (lat / cls / sim) | 4.8 | 2.3 |
| backward, all | 729.3 | 85.5 |
| — CLIP backward (isolated) | ~637 | |
| — cutouts backward (isolated) | 15.1 | 4.0 |
| — BigGAN backward (isolated) | 73.7 | 16.8 |
| Adam + EMA + zero_grad | 2.8 | 1.2 |
| **total** | **1,036.7** | **136.8** |

So CLIP fwd+bwd was **82 %** of a default step, BigGAN fwd+bwd 15 %, and the whole
cutout/loss/optimiser side 3 % (≈ 10 % at 128/8, where the host-side op count — ~4,000
aten calls per step — matters more). The anomaly that pointed at the fix: with frozen
weights, CLIP's backward needs the same matmul FLOPs as its forward (~845 GFLOP at
batch 96), yet took 3× as long.

`scripts/clip_microbench.py 96` split the ViT: one `ResidualAttentionBlock` fwd+bwd is
~29 ms in fp16 (MHA 13, MLP 15, LN 0.7), so twelve blocks are ~350 ms — leaving ~480 ms
of the CLIP backward *outside* the blocks. That was `conv1`, the 32×32/stride-32
patch-embedding conv: `scripts/patch_embed_bench.py 96`:

| patch embedding, batch 96 | fwd | fwd+bwd (input grad) |
|---|---|---|
| `Conv2d(3, 768, 32, 32)` fp16 (the existing "fast" contiguous-grad path) | 20.5 ms | **488.2 ms** |
| same as `reshape → [96, 49, 3072] @ W.T` fp16 | 3.1 ms | **6.3 ms** |
| conv fp32 / matmul fp32 | 22.9 / 3.4 | 612.7 / 6.6 |

Max abs difference conv vs matmul: 1e-3 in fp16 (one ulp at magnitude ~1), 8e-6 in fp32.
That single op was ~47 % of the step. (It is the same conv whose *permuted-with-offset*
grad layout costs 130 s — `docs/pytorch-issues/01-mps-conv-backward-strided-grad-output.md`; the issue draft should gain a line
saying the fast path is still ~80× slower than the equivalent matmul.)

## 2. What changed, ranked by effect (512/96 step, cumulative)

| # | change | where | step after | Δ | numerics vs base |
|---|---|---|---|---|---|
| 1 | Patch embedding as reshape + matmul (MPS only) | `clip.py` `VisualTransformer.patch_embed` | 1,002 → **520** | −482 ms | 1 fp16 ulp |
| 2 | BigGAN: bake spectral norm (174 modules), slice `conv_to_rgb` 128 → 3 out-channels, fuse conditional BN to one `addcmul` | `biggan.py` `freeze_for_inference`, `BigGANBatchNorm.forward` | 520 → **485** | −35 ms | SN bake: **bit-identical** when baked on the device (see §4); rgb slice 1.8e-7; BN 3e-4 on the [-1, 1] image |
| 3 | Fused QuickGELU Metal kernels (fwd + bwd) | `mps_kernels.py`, `clip.py` `QuickGELU` | 485 → **461** | −24 ms | *closer* to fp32 than eager fp16 (bwd err 1.95e-3 vs 8.8e-3) |
| 4 | Fused LayerNorm Metal kernels, fp16 in/out, fp32 statistics, one pass each way | `mps_kernels.py`, `clip.py` `LayerNorm` | 461 → **446** | −15 ms | identical error to the upcast reference (3.89e-3 fwd / 1.95e-3 bwd) |

Isolated numbers behind rows 2–4:

- BigGAN-deep-512, batch 1 (`scripts/biggan_microbench.py 512`): fwd 66.3 → 38.1 ms,
  fwd+bwd 139.0 → 106.6 ms. Of that, spectral-norm bake alone is −5 ms, fused BN −24 ms
  fwd (the 256²/512² blocks are memory-bound: five full passes over a 134 MB activation
  became one); the `conv_to_rgb` slice is the tail 18.8 → 9.5 ms fwd+bwd
  (`scripts/biggan_layers_bench.py 512` — the generator computed 128 channels of a 3×3
  conv at 512² and kept three).
- QuickGELU `[50, 96, 3072]` (`scripts/mps_kernels_test.py`): fp16 fwd+bwd 1.83 → 0.57 ms,
  fp32 3.45 → 0.84 ms. Eager MPS runs the backward as ~6 kernels; fused it is one.
- LayerNorm `[50, 96, 768]` fp16: fwd+bwd 0.76 → 0.20 ms. Native fp16 `F.layer_norm`
  would also be 0.46 ms but with 2× the error (6.7e-3 fwd); the kernel keeps fp32 stats.

Both kernels are compiled at first use with `torch.mps.compile_shader` — no C++/ObjC++
extension, no build step, `half`/`float`/`bfloat` instantiations, wrapped in
`autograd.Function`s with pure-torch references used on every other device and as the
correctness oracle. The LayerNorm kernels use one 256-thread threadgroup per row with
`simd_sum` + threadgroup scratch and save `(mean, rstd)` so the backward is one pass.

Profile after (`scripts/profile_step.py 512 96`): BigGAN fwd 36 · cutouts 9 · CLIP fwd 178
· losses 6 · backward 234 (CLIP 152, cutouts 16, BigGAN 64) · Adam/EMA 1 → 465 ms in the
staged harness, 446 ms in `bench_step.py`.

## 3. What did not help (numbers)

- **dtype.** CLIP already runs in fp16 on MPS — `clip.load` leaves Linear/Conv/proj fp16
  with fp32 LayerNorms, and `encode_image` casts the cutout batch. At batch 96:
  fp16-as-shipped fwd / fwd+bwd 189 / 839 ms; **fp32 201 / 979**; **bf16 272 / 1,254**.
  "Try fp16" was already the baseline; nothing beats it.
- **BigGAN in fp16.** fwd+bwd 107 → 90 ms, but `max|diff| = 0.35` on the [-1, 1] image
  (mean 6e-3) — 17 % of full scale on some pixel. Too much drift for a default. Left out.
- **`torch.compile` (inductor MPS backend, torch 2.14)** on the ViT: 966 vs 979 ms fp32
  fwd+bwd. It fuses a little pointwise work and nothing else on MPS today.
- **Attention rewrites.** Explicit batch-first SDPA 15.3 ms, manual softmax attention
  12.6 ms, `nn.MultiheadAttention` as shipped 13.2 ms per block fwd+bwd — no win, and
  the block's matmuls dominate. Left the module alone.
- **Nearest ×2 upsample alternatives** (BigGAN's 14 upsamples): `F.interpolate` 2.5 ms
  fwd+bwd at `[1,128,256,256]`; `expand+reshape` 287 ms; `repeat_interleave` 127 ms. The
  MPS `upsample_nearest2d` kernel is fine here — the 70 ms of CPU self-time the profiler
  showed against `upsample_nearest2d_backward` was queue back-pressure, not GPU time.
  (Astra separately found its *gradient* is wrong for some crop sizes — a correctness bug,
  script-only repro on `perf/astra`.)
- **Native fp16 LayerNorm** without a kernel: same speed as the Metal kernel, twice the
  error. Not acceptable as a default; superseded by row 4.

## 4. Quality: "same seed → same image" is not on the table, and why

Base-vs-base on MPS is bit-identical run to run (30 steps, image / losses / latents
`max|diff| = 0`), so the harness is trustworthy — and it shows that **any** change of
rounding anywhere gives a different picture for the same seed. Ablations at 128/8 × 30
steps against base (`scripts/parity_dump.py` / `parity_compare.py`):

| variant | step-0 loss | step-1 loss | final image PSNR vs base |
|---|---|---|---|
| all toggles off (harness check) | identical | identical | ∞ (bit-identical) |
| `BIG_SLEEP_REFERENCE_MATH=1` | identical | identical | ∞ (bit-identical) |
| spectral-norm bake, computed on CPU (first version) | identical | −8.750 vs −8.702 | 8.7 dB |
| spectral-norm bake, computed on MPS (shipped) | identical | identical | ∞ (bit-identical) |
| conv_to_rgb slice alone (1.8e-7 fwd diff) | identical | differs | 15.8 dB |
| everything on | identical | −8.773 vs −8.702 | 10.3 dB |

A 1e-7 forward difference or a one-ulp fp16 rounding change is a different image after
~10 steps of Adam at lr 0.07 through a 100×-scaled cosine loss — the optimisation is
chaotic. Astra measured the same thing independently: a 1.1e-8 gradient change in the
latent regulariser produced a visibly different composition after 200 steps.

The spectral-norm bake is a worked example of how fine the line is. My first version baked
`weight_orig / sigma` right after `load_state_dict`, i.e. **on the CPU**, while the live
parametrisation computes that same division **on MPS** every forward; the two round
differently by an ulp on some weights, and 30 steps later the image was unrecognisable
(8.7 dB). A per-layer probe (`scripts/sn_bake_grad_probe.py`) showed the baked weight,
forward and input-gradient bit-identical once the bake ran on the device, and the full
30-step run confirms it: baking on MPS is exact, so it is always on — the reference path
included — and `from_pretrained` moves the model to the device before baking for that
reason.

So the bar becomes *equivalent quality*, and the evidence is the end-to-end run:
`dream "a pyramid made of ice" --fast` (seed 0, 200 steps) —
`samples/mps_fast_a_pyramid_made_of_ice.png` (base, 153 s) versus
`samples/perf_fable_fast_a_pyramid_made_of_ice.png` (this branch, 76 s). Both are
unmistakably an ice pyramid; different compositions; final losses over 30 steps at
128/8 are −23.3 (base) vs −25.2 (branch), i.e. the branch is not optimising worse. The
two Metal kernels are strictly *more* accurate than the eager fp16 ops they replace.

Anyone who needs the old images back for a seed: `BIG_SLEEP_REFERENCE_MATH=1`.

## 5. Where the floor is now

446 ms at 512/96 breaks down as roughly CLIP 330 (fwd 178 + bwd 152), BigGAN 100,
cutouts 25, everything else 10.

- **CLIP is now matmul-bound.** Twelve blocks × 136 GFLOP fwd+bwd(input) = 1.63 TFLOP
  per step in ~330 ms ≈ 5 TFLOPS achieved. Big fp16 GEMMs on this GPU through MPS top out
  around 8–9 TFLOPS, so the matmul-only floor is ~190–200 ms; the ~130 ms above it is
  MPS's GEMM efficiency on `[4800, 768] × [768, 3072]`-class shapes and the attention's
  many small `[1152, 50, 64]` bmms. Beating MPS's GEMM with a hand-written Metal kernel is
  a project, not an afternoon; I would not expect more than ~10–15 % from it.
- **BigGAN** (~100 ms) is memory-bound in the 256²/512² blocks. A fused
  BN-affine + ReLU kernel with the per-channel reductions its backward needs (grad w.r.t.
  the conditioning scale/offset) is the next hand-rolled candidate: maybe −10 to −15 ms.
  fp16 would take another ~17 ms but costs visible drift (§3).
- **Cutouts** (~25 ms fwd+bwd) are Astra's lane: its Metal cutout kernel measures
  19.1 → 2.0 ms fwd+bwd at 512/96 but is script-only because it also *fixes* the MPS
  nearest-backward gradient bug and therefore changes trajectories — the same parity
  argument as §4 applies, so it could ship on the same terms as this branch.
- **Host overhead** dominates at 128/8 (64 ms for ~40 ms of GPU work): ~4,000 aten calls
  per step. CUDA-graph-style capture does not exist for MPS; fewer, larger ops (Astra's
  regulariser vectorisation, which it withdrew for drift) is the only lever.

Realistic floor with everything above shipped: **~350–380 ms at 512/96** (2.7× base),
with the last ~150 ms to the theoretical GEMM limit needing a better GEMM than MPS's.

## 6. Reproducing

```bash
source ~/dev/experimental/big-sleep-mps/.venv/bin/activate
cd <this checkout>; export PYTHONPATH=$PWD      # the venv's editable install points at another checkout
python scripts/bench_step.py 512 96 --n 20
python scripts/profile_step.py 512 96           # stage breakdown; --ops adds the aten table
python scripts/clip_microbench.py 96 --compile  # dtype / block / attention / compile probes
python scripts/patch_embed_bench.py 96
python scripts/biggan_microbench.py 512; python scripts/biggan_layers_bench.py 512
python scripts/mps_kernels_test.py              # Metal kernels vs reference vs fp32 oracle
python scripts/parity_dump.py 128 8 30 /tmp/a.pt [--no-patch-embed --no-gelu --no-ln --no-fused-bn --no-rgb-slice]
python scripts/parity_compare.py /tmp/a.pt /tmp/b.pt
BIG_SLEEP_REFERENCE_MATH=1 python scripts/bench_step.py 512 96   # the exact, slow path
```

Gotchas met on the way: the venv's editable install resolves `big_sleep` to
`~/dev/experimental/big-sleep-mps`, not the worktree (Astra caught my first profile
running the 128 s path); `copy.deepcopy` of an MPS module aliases parameter storage, so
`.to(dtype)` on the copy mutates the original — reload instead; and a fresh `nn.Conv2d`
built inside model loading consumes the CPU RNG before the seeded latents are drawn (the
`conv_to_rgb` slice is done in place for that reason).
