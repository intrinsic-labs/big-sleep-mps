"""Time one optimisation step of Big Sleep at a given image size / cutout count.

Usage:
    python scripts/bench_step.py 512 96          # CLI defaults
    python scripts/bench_step.py 512 64          # --fast cutouts
    python scripts/bench_step.py 128 8 --n 3     # smoke-sized

Prints the mean wall time of `Imagine.train_step` after one warm-up step,
with the device synchronised around the timed region. Weights are downloaded
on first run (~800 MB total for BigGAN-512 + CLIP ViT-B/32).
"""
import argparse
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_size", type=int)
    ap.add_argument("num_cutouts", type=int)
    ap.add_argument("--n", type=int, default=2, help="timed steps after warm-up")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]  # keep fire/argparse in the library quiet

    from big_sleep.big_sleep import Imagine
    from big_sleep.device import DEVICE, synchronize

    im = Imagine(
        text="a pyramid made of ice", image_size=args.image_size,
        num_cutouts=args.num_cutouts, epochs=1, iterations=1,
        save_every=10**9, open_folder=False, seed=1,
    )
    im.train_step(0, 0)
    synchronize()
    t = time.perf_counter()
    for _ in range(args.n):
        im.train_step(0, 0)
    synchronize()
    per_step = (time.perf_counter() - t) / args.n
    print(f"device={DEVICE} torch={torch.__version__} "
          f"image_size={args.image_size} num_cutouts={args.num_cutouts} "
          f"train_step={per_step*1000:.0f} ms")


if __name__ == "__main__":
    main()
