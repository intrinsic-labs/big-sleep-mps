# big-sleep-mps

A Mac port of [Big Sleep](https://github.com/lucidrains/big-sleep) — Ryan Murdock's
CLIP + BigGAN text-to-image technique, packaged by Phil Wang (lucidrains) — that runs
on Apple Silicon through PyTorch's MPS backend. It is 2021-era "dream" imagery, not a
diffusion model; think of it as a retro text-to-image toy that now runs at about two
steps per second on a MacBook instead of one step every two minutes.

<img src="./samples/perf_fast_a_pyramid_made_of_ice.png" width="256px"></img> <img src="./samples/metal_cutouts_fast_a_pyramid_made_of_ice.png" width="256px"></img> <img src="./samples/a_pyramid_made_of_ice.png" width="256px"></img>

*Left and middle: "a pyramid made of ice" from this fork, `dream "a pyramid made of ice" --fast` on an M1 Max — 200 steps, about 75 seconds wall including model load — on the default path and with `--metal_cutouts` (same seed; see below on why they differ). Right: upstream's sample for the same prompt (V100, 20 × 1050 steps). `samples/mps_fast_a_pyramid_made_of_ice.png` is the same run on the pre-optimisation tree; the other images in `samples/` are upstream's.*

## Install

This fork is **not on PyPI**. `pip install big-sleep` installs lucidrains' original
package, which asserts CUDA at import and cannot run on a Mac. Install from git:

```bash
pip install git+https://github.com/intrinsic-labs/big-sleep-mps
```

Requires Python 3.9+ and PyTorch 2.0+ (installed for you). Weights download on first
run to `~/.cache/clip` and `~/.pytorch_pretrained_biggan` (~800 MB for the 512 px model).
CUDA and CPU still work; the device is picked automatically (MPS → CUDA → CPU).

## Usage

```bash
dream "a pyramid made of ice"                    # 512 px, 1 × 500 steps, 96 cutouts
dream "a pyramid made of ice" --fast             # 1 × 200 steps, 64 cutouts
dream "a pyramid made of ice" --upstream_defaults  # lucidrains' 20 × 1050 schedule
dream "a pyramid made of ice" --output_dir out --seed 42
dream "a pyramid made of ice" --metal_cutouts      # MPS: fused cutout kernel, see docs/performance.md
```

Images are saved to the current directory (or `--output_dir`). The best-scoring image
so far is written alongside as `<name>.best.png` (`--save_best=False` to disable).
Multiple prompts are separated by `|`; `--text_min "blur|zoom"` penalises phrases.
`--seed N` reproduces a run; `--random` picks a fresh seed. `python -m big_sleep.cli --help`
lists everything.

The Python API is unchanged from upstream:

```python
from big_sleep import Imagine
Imagine(text="fire in the sky", lr=5e-2, save_every=25, save_progress=True)()
```

## Why it was slow, and how fast it is now

The original port ran, but one optimisation step at the CLI defaults took ~128 s on an M1 Max —
the documented default run would have taken a month. Two passes fixed that. The first found the
whole cost in one MPS kernel: `mps_convolution_backward` on CLIP's 32×32/stride-32
patch-embedding conv is ~200× slower when the incoming gradient is a permuted view with a storage
offset, which is what autograd hands it because CLIP prepends the class token with `torch.cat`;
a `.contiguous()` hook fixes it, and freezing the CLIP/BigGAN weights takes another third off.
The second pass profiled what was left and took it down another 2.2×: the patch embedding as a
matmul (that conv is ~80× slower than the equivalent GEMM on MPS even on its fast path), a baked
BigGAN, and hand-written Metal kernels for QuickGELU, LayerNorm and (opt-in) the cutouts.

| step (M1 Max 32 GB, torch 2.14.0) | original port | PR #1 | **now** | `--metal_cutouts` |
|---|---|---|---|---|
| 512 px, 96 cutouts (defaults) | 128,201 ms | 1,002 ms | **447 ms** | 435 ms |
| 512 px, 64 cutouts (`--fast`) | 86,232 ms | 721 ms | **331 ms** | 327 ms |
| 128 px, 8 cutouts | 10,784 ms | 121 ms | **62 ms** | 63 ms |
| `dream "…" --fast`, wall incl. model load | — | 153 s | **76 s** | 75 s |

Two things to know. **Same seed, different image**: the optimisation is chaotic enough that any
change of rounding (one fp16 ulp) gives a different — equivalent — picture after ~10 steps, so
the optimised path does not reproduce the old images; `BIG_SLEEP_REFERENCE_MATH=1` restores the
upstream arithmetic bit for bit at the old speed. And `--metal_cutouts` is off by default because
it *corrects* an MPS gradient bug (torch's nearest-resize backward hits the wrong pixels for some
sizes) and therefore changes trajectories too. The whole story, the profile, what failed, and the
floor: [`docs/performance.md`](docs/performance.md). `scripts/bench_step.py` times a step on your
own machine; three ready-to-file PyTorch issue drafts are under
[`docs/pytorch-issues/`](docs/pytorch-issues/).

## What this fork changes

- Device selection (MPS / CUDA / CPU) in one place, `big_sleep/device.py`.
- The MPS speed work above: contiguous-grad hook, frozen weights, matmul patch embedding,
  baked BigGAN, Metal QuickGELU/LayerNorm kernels (all MPS-only; CUDA/CPU numerics match
  upstream), the opt-in Metal cutout kernel, and `BIG_SLEEP_REFERENCE_MATH=1`.
- Mac-sized CLI defaults (`--upstream_defaults` restores the originals), `--fast`,
  `--output_dir`, `--debug`, and a working `--seed` (Python's `random`, which picks the
  cutout offsets, is now seeded too) and `--torch_deterministic`.
- A CPU smoke test in CI. No PyPI publishing.

## Credits and license

All of the interesting ideas are Ryan Murdock's ([@advadnoun](https://twitter.com/advadnoun));
the package this fork is built on is [lucidrains/big-sleep](https://github.com/lucidrains/big-sleep),
whose git history is preserved here. CLIP is OpenAI's; BigGAN-deep weights are via
[pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN).
MIT License — upstream copyright retained, see [LICENSE](LICENSE).

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
@misc{brock2019large,
    title   = {Large Scale GAN Training for High Fidelity Natural Image Synthesis},
    author  = {Andrew Brock and Jeff Donahue and Karen Simonyan},
    year    = {2019},
    eprint  = {1809.11096},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
