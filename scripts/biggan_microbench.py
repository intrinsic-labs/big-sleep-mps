"""Where BigGAN-deep's forward/backward goes on MPS, and what baking helps.

    PYTHONPATH=$PWD python scripts/biggan_microbench.py 512
"""
import sys, time, math, collections
import torch, torch.nn as nn, torch.nn.functional as F

size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = "mps"


def sync(): torch.mps.synchronize()


def bench(label, fn, n=5, warm=2):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); ms = (time.perf_counter() - t) / n * 1000
    print(f"{label:62s} {ms:9.1f} ms"); return ms


from big_sleep.biggan import BigGAN, BigGANBatchNorm, SelfAttn, GenBlock
from big_sleep.big_sleep import Latents

gan = BigGAN.from_pretrained(f"biggan-deep-{size}").to(dev).eval().requires_grad_(False)
lat = Latents(num_latents=len(gan.config.layers) + 1, num_classes=gan.config.num_classes, z_dim=gan.config.z_dim).to(dev)
torch.manual_seed(0)


def fwd():
    with torch.no_grad():
        return gan(*lat(), 1)


def fwd_bwd():
    out = gan(*lat(), 1)
    out.sum().backward()
    lat.zero_grad()


print(f"== BigGAN-deep-{size}, batch 1, torch {torch.__version__} ==")
ref = fwd().clone()
bench("as shipped: fwd (no_grad)", fwd)
bench("as shipped: fwd+bwd (to latents)", fwd_bwd)

# per-module-type forward accounting (sync in hooks -> inflated absolute, honest relative)
acc = collections.Counter(); starts = {}
def pre(m, i): sync(); starts[m] = time.perf_counter()
def post(m, i, o): sync(); acc[type(m).__name__] += time.perf_counter() - starts[m]
hs = []
for m in gan.modules():
    if len(list(m.children())) == 0 or isinstance(m, BigGANBatchNorm):
        hs += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
fwd(); acc.clear(); fwd()
for h in hs: h.remove()
print("  forward by module type (ms, with per-module syncs):")
for k, v in acc.most_common(): print(f"    {k:28s} {v*1000:7.1f}")

# 1) bake spectral norm
n_sn = 0
for m in list(gan.modules()):
    for k, hook in list(m._forward_pre_hooks.items()):
        if type(hook).__name__ == "SpectralNorm":
            nn.utils.remove_spectral_norm(m, name=hook.name); n_sn += 1
gan.requires_grad_(False)
print(f"  removed spectral_norm from {n_sn} modules; max|out - ref| = {(fwd() - ref).abs().max().item():.3e}")
bench("baked SN: fwd (no_grad)", fwd)
bench("baked SN: fwd+bwd (to latents)", fwd_bwd)

# 2) fused conditional BN affine: (x - m)/sqrt(v+eps)*w + b  ->  x*a + b'
def bn_fused_forward(self, x, truncation, condition_vector=None):
    coef, start_idx = math.modf(truncation / self.step_size); start_idx = int(start_idx)
    if coef != 0.0:
        running_mean = self.running_means[start_idx] * coef + self.running_means[start_idx + 1] * (1 - coef)
        running_var = self.running_vars[start_idx] * coef + self.running_vars[start_idx + 1] * (1 - coef)
    else:
        running_mean, running_var = self.running_means[start_idx], self.running_vars[start_idx]
    if self.conditional:
        inv = torch.rsqrt(running_var + self.eps)
        weight = (1 + self.scale(condition_vector)) * inv          # [1, C]
        bias = self.offset(condition_vector) - running_mean * weight
        return torch.addcmul(bias[:, :, None, None], x, weight[:, :, None, None])
    return F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=False, momentum=0.0, eps=self.eps)
BigGANBatchNorm.forward = bn_fused_forward
print(f"  fused BN affine; max|out - ref| = {(fwd() - ref).abs().max().item():.3e}")
bench("baked SN + fused BN: fwd (no_grad)", fwd)
bench("baked SN + fused BN: fwd+bwd (to latents)", fwd_bwd)

# 3) fp16 generator
gan.half()
def fwd16():
    with torch.no_grad():
        z, c = lat(); return gan(z.half(), c.half(), 1)
def fwd_bwd16():
    z, c = lat(); out = gan(z.half(), c.half(), 1); out.float().sum().backward(); lat.zero_grad()
o16 = fwd16().float()
print(f"  fp16 generator; max|out - ref| = {(o16 - ref).abs().max().item():.3e}  mean|diff| = {(o16 - ref).abs().mean().item():.3e}  finite={torch.isfinite(o16).all().item()}")
bench("baked SN + fused BN + fp16: fwd (no_grad)", fwd16)
bench("baked SN + fused BN + fp16: fwd+bwd (to latents)", fwd_bwd16)
