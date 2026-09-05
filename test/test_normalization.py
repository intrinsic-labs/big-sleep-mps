"""Fixed CLIP normalization must preserve pixels and input gradients exactly."""
import pytest
import torch

from big_sleep.clip import _transform
from big_sleep.normalization import normalize_clip_image


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_normalization_matches_torchvision(device, dtype):
    if device == "mps" and (not torch.backends.mps.is_available() or dtype == torch.float64):
        pytest.skip("This device/dtype is unavailable")
    torch.manual_seed(7)
    image = torch.randn(2, 3, 9, 7, dtype=dtype).to(device)
    image = image.transpose(-1, -2).requires_grad_()
    before = image.detach().clone()
    reference = _transform()(image)
    actual = normalize_clip_image(image)
    grad = torch.randn_like(image)
    expected_grad, = torch.autograd.grad(reference, image, grad)
    actual_grad, = torch.autograd.grad(actual, image, grad)
    assert torch.equal(image, before)
    assert torch.equal(actual, reference)
    assert torch.equal(actual_grad, expected_grad)
