"""Exercise every default crop size against the exact forward-index oracle."""
import pytest
import torch
import torch.nn.functional as F

from big_sleep.mps_cutouts import metal_cutouts


@pytest.mark.skipif(not torch.backends.mps.is_available()
                    or not hasattr(torch.mps, "compile_shader"),
                    reason="Metal cutouts require MPS and compile_shader")
@pytest.mark.parametrize("width", [128, 256, 512])
def test_all_default_sizes_match_source_indices_and_gradient_counts(width):
    # Integer-valued fp32 pixels make both forward identities and the derivative
    # of sum(output) exact, independent of any backend's resize backward.
    cpu = torch.arange(3 * width * width, dtype=torch.float32).reshape(1, 3, width, width)
    boxes = [(size, (width - size) // 2, width - size)
             for size in range(int(width * .5), int(width * .95) + 1)]
    image = cpu.to("mps").requires_grad_()
    metadata = torch.tensor(boxes, dtype=torch.int32, device="mps")
    actual = metal_cutouts(image, metadata)
    actual.sum().backward()
    output = actual.detach().cpu()
    expected_grad = torch.zeros(cpu.numel(), dtype=torch.int64)
    for index, (size, row, col) in enumerate(boxes):
        expected = F.interpolate(cpu[:, :, row:row + size, col:col + size],
                                 (224, 224), mode="nearest")[0]
        assert torch.equal(output[index], expected), f"forward mismatch at crop size {size}"
        expected_grad += torch.bincount(expected.long().flatten(), minlength=cpu.numel())
    assert torch.equal(image.grad.cpu(), expected_grad.reshape_as(cpu).float())
