"""MPS nearest-resize backward can send gradients to different input pixels.

Run: python scripts/mps_nearest_backward_repro.py
No models, weights, Big Sleep imports, or random numbers are required.
The pixel values encode their source indices, so bincount of the actual forward
output is an independent exact derivative oracle for sum(resize(input)).
"""
import argparse

import torch
import torch.nn.functional as F


def check(size, target):
    cpu = torch.arange(size * size, dtype=torch.float32).reshape(1, 1, size, size)
    mps = cpu.to("mps").requires_grad_()
    expected_output = F.interpolate(cpu, (target, target), mode="nearest")
    actual_output = F.interpolate(mps, (target, target), mode="nearest")
    actual_output.sum().backward()
    output = actual_output.detach().cpu()
    expected_grad = torch.bincount(output.long().flatten(), minlength=size * size)
    expected_grad = expected_grad.reshape_as(cpu).float()
    actual_grad = mps.grad.cpu()
    error = (expected_grad - actual_grad).abs()
    print(f"{size} -> {target}: forward_equal={torch.equal(output, expected_output)} "
          f"wrong_gradient_pixels={torch.count_nonzero(error).item()} "
          f"max_gradient_error={error.max().item():g}")
    if error.any():
        row, col = error[0, 0].nonzero()[0].tolist()
        print(f"  first mismatch ({row}, {col}): "
              f"expected={expected_grad[0, 0, row, col].item():g}, "
              f"MPS={actual_grad[0, 0, row, col].item():g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[104, 304, 348, 390, 454, 468, 409])
    parser.add_argument("--target", type=int, default=224)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("An MPS-capable Python process is required.")
    print(f"torch={torch.__version__} git={torch.version.git_version}")
    for size in args.sizes:
        check(size, args.target)


if __name__ == "__main__":
    main()
