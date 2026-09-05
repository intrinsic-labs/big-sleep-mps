"""Per-layer fwd / fwd+bwd of BigGAN-deep-512 (after SN bake + fused BN), plus targeted op probes.

    PYTHONPATH=$PWD python scripts/biggan_layers_bench.py 512
"""
import sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F

size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = "mps"


def sync(): torch.mps.synchronize()


def bench(label, fn, n=5, warm=2, quiet=False):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); ms = (time.perf_counter() - t) / n * 1000
    if not quiet: print(f"{label:66s} {ms:9.1f} ms")
    return ms


from big_sleep.biggan import BigGAN, BigGANBatchNorm, GenBlock, SelfAttn
from big_sleep.big_sleep import Latents

gan = BigGAN.from_pretrained(f"biggan-deep-{size}").to(dev).eval().requires_grad_(False)
print("layers (up, in_mult, out_mult):", gan.config.layers, "ch", gan.config.channel_width, "attn@", gan.config.attention_layer_position)
for m in list(gan.modules()):
    for k, hook in list(m._forward_pre_hooks.items()):
        if type(hook).__name__ == "SpectralNorm":
            nn.utils.remove_spectral_norm(m, name=hook.name)
gan.requires_grad_(False)

def bn_fused_forward(self, x, truncation, condition_vector=None):
    coef, start_idx = math.modf(truncation / self.step_size); start_idx = int(start_idx)
    running_mean, running_var = self.running_means[start_idx], self.running_vars[start_idx]
    if self.conditional:
        inv = torch.rsqrt(running_var + self.eps)
        weight = (1 + self.scale(condition_vector)) * inv
        bias = self.offset(condition_vector) - running_mean * weight
        return torch.addcmul(bias[:, :, None, None], x, weight[:, :, None, None])
    return F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=False, momentum=0.0, eps=self.eps)
BigGANBatchNorm.forward = bn_fused_forward

lat = Latents(num_latents=len(gan.config.layers) + 1, num_classes=gan.config.num_classes, z_dim=gan.config.z_dim).to(dev)
z, c = lat()
embed = gan.embeddings(c); cond = torch.cat((z, embed), dim=1).detach()

# capture each top-level generator layer's input
inputs = {}
hs = [m.register_forward_pre_hook(lambda m, i, m_=m: inputs.__setitem__(m_, i[0].detach())) for m in gan.generator.layers]
with torch.no_grad(): gan.generator(cond, 1)
for h in hs: h.remove()

print(f"\n== per layer, fwd / fwd+bwd(input+cond) ==")
tot_f = tot_fb = 0
idx = 1
for i, layer in enumerate(gan.generator.layers):
    x = inputs[layer]
    if isinstance(layer, GenBlock):
        cv = cond[idx].unsqueeze(0); idx += 1
        f = lambda: layer(x, cv, 1)
        def fb():
            xi = x.clone().requires_grad_(); cvi = cv.clone().requires_grad_()
            layer(xi, cvi, 1).sum().backward()
        label = f"GenBlock[{i}] in{tuple(x.shape)} up={layer.up_sample}"
    else:
        f = lambda: layer(x)
        def fb():
            xi = x.clone().requires_grad_(); layer(xi).sum().backward()
        label = f"SelfAttn[{i}] in{tuple(x.shape)}"
    with torch.no_grad(): mf = bench("", f, quiet=True)
    mfb = bench("", fb, quiet=True)
    tot_f += mf; tot_fb += mfb
    print(f"  {label:58s} fwd {mf:6.1f}  fwd+bwd {mfb:6.1f} ms")
x = inputs[gan.generator.layers[-1]]
with torch.no_grad(): xo = gan.generator.layers[-1](x, cond[idx - 1].unsqueeze(0), 1).detach()
g = gan.generator
def tail(xi): return torch.tanh(g.conv_to_rgb(torch.relu(g.bn(xi, 1)))[:, :3])
def tail_fb():
    xi = xo.clone().requires_grad_(); tail(xi).sum().backward()
with torch.no_grad(): mf = bench("", lambda: tail(xo), quiet=True)
mfb = bench("", tail_fb, quiet=True)
print(f"  {'tail: bn+relu+conv_to_rgb(128->128 3x3)+[:3]+tanh':58s} fwd {mf:6.1f}  fwd+bwd {mfb:6.1f} ms")
tot_f += mf; tot_fb += mfb
print(f"  {'SUM':58s} fwd {tot_f:6.1f}  fwd+bwd {tot_fb:6.1f} ms")

# conv_to_rgb sliced to the 3 channels actually used
w3 = g.conv_to_rgb.weight[:3].clone(); b3 = g.conv_to_rgb.bias[:3].clone()
def tail3(xi): return torch.tanh(F.conv2d(torch.relu(g.bn(xi, 1)), w3, b3, padding=1))
def tail3_fb():
    xi = xo.clone().requires_grad_(); tail3(xi).sum().backward()
with torch.no_grad(): print(f"  conv_to_rgb sliced to 3 out-ch: max|diff| = {(tail3(xo) - tail(xo)).abs().max().item():.2e}")
with torch.no_grad(): bench("  tail with conv_to_rgb[:3] fwd", lambda: tail3(xo))
bench("  tail with conv_to_rgb[:3] fwd+bwd", tail3_fb)

# upsample_nearest x2 probes at the biggest GenBlock shapes
print("\n== nearest x2 upsample variants (fwd+bwd) ==")
for shape in ((1, 128, 256, 256), (1, 256, 128, 128)):
    a = torch.randn(*shape, device=dev)
    def up_interp(t): return F.interpolate(t, scale_factor=2, mode="nearest")
    def up_expand(t):
        n, ch, h, w = t.shape
        return t[:, :, :, None, :, None].expand(n, ch, h, 2, w, 2).reshape(n, ch, 2 * h, 2 * w)
    def up_repeat(t): return t.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    for name, fn in (("F.interpolate nearest", up_interp), ("expand+reshape", up_expand), ("repeat_interleave", up_repeat)):
        def fb():
            ti = a.clone().requires_grad_(); fn(ti).sum().backward()
        with torch.no_grad(): assert torch.equal(fn(a), up_interp(a))
        bench(f"  {shape} {name}", fb)
