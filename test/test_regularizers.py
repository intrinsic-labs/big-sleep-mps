"""Compare both values and gradients against the pre-optimization prior."""
import pytest
import torch

from big_sleep.regularizers import latent_loss


def reference_latent_loss(latents, threshold):
    loss = (
        (1 - latents.std(dim=1)).abs().mean()
        + latents.mean(dim=1).abs().mean()
        + 4 * torch.maximum(latents.square().mean(), threshold)
    )
    for row in latents:
        diffs = row - row.mean()
        zscores = diffs / diffs.pow(2).mean().pow(0.5)
        loss = loss + (zscores.pow(4).mean() - 3).abs() / len(latents)
        loss = loss + zscores.pow(3).mean().abs() / len(latents)
    return loss


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("count", [11, 13, 15])
@pytest.mark.parametrize("scale", [0.3, 1.7])
def test_batched_moments_preserve_value_and_gradient(device, count, scale):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    torch.manual_seed(count)
    latents = (torch.randn(count, 128) * scale + 0.1).to(device).requires_grad_()
    threshold = torch.tensor(1, device=device)
    reference = reference_latent_loss(latents, threshold)
    actual = latent_loss(latents, threshold)
    expected_grad, = torch.autograd.grad(reference, latents)
    actual_grad, = torch.autograd.grad(actual, latents)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)
