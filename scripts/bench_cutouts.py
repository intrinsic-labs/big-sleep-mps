"""Compare reference, indexed, and experimental Metal nearest cutouts.

Run: python scripts/bench_cutouts.py 512 96 --n 30
Reports a CPU-gradient oracle and synchronized forward/backward means. The
resampled measurement includes geometry construction and host-to-device copies.
The Metal kernel is the opt-in runtime path (BIG_SLEEP_METAL_CUTOUTS=1 /
--metal_cutouts): its corrected gradients change the MPS optimization trajectory
for a seed. Requires torch.mps.compile_shader.
"""
import argparse
import random
import time

import torch
import torch.nn.functional as F

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from big_sleep.mps_cutouts import metal_cutouts  # noqa: E402


def sample_boxes(width, count):
    boxes = []
    for _ in range(count):
        size = int(width * torch.zeros(1).normal_(mean=.8, std=.3).clip(.5, .95))
        boxes.append((size, random.randint(0, width - size), random.randint(0, width - size)))
    return boxes


def reference(image, boxes):
    return torch.cat([
        F.interpolate(image[:, :, row:row + size, col:col + size], (224, 224), mode="nearest")
        for size, row, col in boxes
    ])


def index_tensor(boxes, width, device):
    position = torch.arange(224, dtype=torch.float32)
    indices = []
    for size, row, col in boxes:
        local = (position * (size / 224)).floor().long()
        indices.append((local[:, None] + row) * width + local[None, :] + col)
    return torch.stack(indices).to(device)


def gathered(image, indices):
    return image.reshape(3, -1)[:, indices.reshape(-1)].reshape(3, -1, 224, 224).permute(1, 0, 2, 3).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("width", type=int, choices=[128, 256, 512])
    parser.add_argument("count", type=int)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(1)
    random.seed(1)
    boxes = sample_boxes(args.width, args.count)
    cpu = torch.randn(1, 3, args.width, args.width, requires_grad=True)
    cpu_out = reference(cpu, boxes)
    cpu_grad = torch.randn_like(cpu_out)
    expected_grad, = torch.autograd.grad(cpu_out, cpu, cpu_grad)
    image = cpu.detach().to("mps").requires_grad_()
    grad = cpu_grad.to("mps")
    indices = index_tensor(boxes, args.width, image.device)
    metadata = torch.tensor(boxes, dtype=torch.int32, device=image.device)
    fixed = {"torch": lambda: reference(image, boxes),
             "indexed": lambda: gathered(image, indices),
             "metal": lambda: metal_cutouts(image, metadata)}
    for name, fn in fixed.items():
        out = fn()
        actual_grad, = torch.autograd.grad(out, image, grad)
        print(f"{name}: pixel_error={(out.detach().cpu() - cpu_out.detach()).abs().max().item():g} "
              f"gradient_error={(actual_grad.cpu() - expected_grad).abs().max().item():g}", flush=True)

    def measure(name, fn):
        def step():
            image.grad = None
            fn().backward(grad)
        for _ in range(3):
            step()
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(args.n):
            step()
        torch.mps.synchronize()
        print(f"{name}: {(time.perf_counter() - start) * 1000 / args.n:.3f} ms", flush=True)

    for name, fn in fixed.items():
        measure(name + " fixed boxes", fn)
    for name in fixed:
        torch.manual_seed(1)
        random.seed(1)
        def resampled():
            coordinates = sample_boxes(args.width, args.count)
            if name == "torch":
                return reference(image, coordinates)
            if name == "indexed":
                return gathered(image, index_tensor(coordinates, args.width, image.device))
            meta = torch.tensor(coordinates, dtype=torch.int32, device=image.device)
            return metal_cutouts(image, meta)
        measure(name + " resampled boxes", resampled)


if __name__ == "__main__":
    main()
