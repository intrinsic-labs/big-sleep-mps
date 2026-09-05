"""Rejected vectorized prior: tiny MPS gradient drift changes seeded images.

Forward/backward was 3.119 -> 0.709 ms on M1 Max, but even a 1.12e-8 gradient
difference changed the seed-0 composition over 200 steps. Kept for research;
BigSleep intentionally retains the original per-row computation.
"""
import torch


def latent_loss(latents, threshold):
    """Upstream's normal-distribution prior, with batched moment reductions.

    Keep the sample standard deviation in the scale penalty, the population
    standard deviation in the moments, and the original scalar addition order.
    These details matter when optimizing the same seed through two networks.
    """
    loss = (
        (1 - latents.std(dim=1)).abs().mean()
        + latents.mean(dim=1).abs().mean()
        + 4 * torch.maximum(latents.square().mean(), threshold)
    )
    diffs = latents - latents.mean(dim=1, keepdim=True)
    std = diffs.pow(2).mean(dim=1, keepdim=True).pow(0.5)
    zscores = diffs / std
    count = latents.shape[0]
    kurtoses = (zscores.pow(4).mean(dim=1) - 3).abs() / count
    skews = zscores.pow(3).mean(dim=1).abs() / count
    for kurtosis, skew in zip(kurtoses.unbind(), skews.unbind()):
        loss = loss + kurtosis + skew
    return loss
