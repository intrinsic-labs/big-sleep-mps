"""Time actual graph boundaries, including random cutout backward, on MPS.

Run: python scripts/profile_pipeline.py 512 96 --n 20 --ops
Synchronizing boundaries perturbs execution: use bench_step.py for whole-step
comparisons. CPU profiler events are dispatch/wait times, not Metal kernel times.
Run this and other GPU benchmarks serially, in separate processes.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import torch

# A sibling editable install must never silently select a different checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_size", type=int)
    parser.add_argument("num_cutouts", type=int)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--ops", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSON phase means")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    from big_sleep.big_sleep import Imagine
    from big_sleep.device import DEVICE, synchronize

    im = Imagine(text="a pyramid made of ice", image_size=args.image_size,
                 num_cutouts=args.num_cutouts, save_every=10**9, open_folder=False, seed=1)
    im.train_step(0, 0)
    synchronize()
    events = []
    measurements = defaultdict(list)

    def mark(label):
        synchronize()
        events.append((label, time.perf_counter()))

    def gradient_boundary(label):
        def hook(grad):
            mark(label)
            return grad
        return hook

    generator_forward = im.model.model.forward
    encode_image = im.model.perceptor.encode_image

    def generator(*a, **kw):
        mark("generator start")
        out = generator_forward(*a, **kw)
        mark("generator end")
        out.register_hook(gradient_boundary("cutouts backward end"))
        return out

    def encode(image):
        mark("CLIP start")
        image.register_hook(gradient_boundary("CLIP backward end"))
        embedding = encode_image(image)
        embedding.register_hook(gradient_boundary("loss backward end"))
        mark("CLIP end")
        return embedding

    im.model.model.forward = generator
    im.model.perceptor.encode_image = encode
    for _ in range(args.n):
        events.clear()
        out, losses = im.model(im.encoded_texts["max"], im.encoded_texts["min"])
        loss = sum(losses)
        mark("backward start")
        loss.backward()
        mark("backward end")
        im.optimizer.step()
        mark("Adam end")
        im.model.model.latents.update()
        mark("EMA end")
        im.optimizer.zero_grad()
        mark("zero grad end")
        for (start, t), (end, u) in zip(events, events[1:]):
            measurements[f"{start} -> {end}"].append((u - t) * 1000)
        del out, losses, loss
    im.model.model.forward = generator_forward
    im.model.perceptor.encode_image = encode_image
    result = {name: sum(times) / len(times) for name, times in measurements.items()}
    print(json.dumps(result, indent=2))
    if DEVICE.type == "mps":
        print(f"allocated={torch.mps.current_allocated_memory() / 1e6:.1f} MB "
              f"driver={torch.mps.driver_allocated_memory() / 1e6:.1f} MB")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    # record_shapes can retain tensors; profile only after the phase measurements.
    if args.ops:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True) as profile:
            im.train_step(0, 0)
            synchronize()
        print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=35))


if __name__ == "__main__":
    main()
