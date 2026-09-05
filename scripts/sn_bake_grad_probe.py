"""Is baking spectral norm bit-exact? Forward yes; this checks the backward, per layer type.
    PYTHONPATH=$PWD python scripts/sn_bake_grad_probe.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import big_sleep.biggan as bg
dev = "mps"
torch.manual_seed(0)
# one snconv2d layer: SN-parametrized vs baked, same weights, same input -> compare input grads
conv_sn = bg.snconv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, eps=1e-12).to(dev).eval().requires_grad_(False)
x = torch.randn(1, 32, 512, 512, device=dev)
g = torch.randn(1, 32, 512, 512, device=dev)
xi = x.clone().requires_grad_(); y_sn = conv_sn(xi); y_sn.backward(g); g_sn = xi.grad.clone()
w_sn = conv_sn.weight.detach().clone()
nn.utils.remove_spectral_norm(conv_sn); conv_sn.requires_grad_(False)
print("baked weight identical bits:", torch.equal(w_sn, conv_sn.weight.detach()), "weight contiguous:", conv_sn.weight.is_contiguous(), w_sn.is_contiguous())
xi = x.clone().requires_grad_(); y_b = conv_sn(xi); y_b.backward(g); g_b = xi.grad.clone()
print("conv3x3 [1,32,512,512]: fwd identical:", torch.equal(y_sn, y_b), " input-grad identical:", torch.equal(g_sn, g_b), " max|dgrad|:", (g_sn - g_b).abs().max().item())
# same, plain F.conv2d with the weight as a fresh temporary vs a Parameter
wt = conv_sn.weight.detach()
xi = x.clone().requires_grad_(); F.conv2d(xi, wt * 1.0, conv_sn.bias, padding=1).backward(g); g_tmp = xi.grad.clone()
xi = x.clone().requires_grad_(); F.conv2d(xi, wt, conv_sn.bias, padding=1).backward(g); g_par = xi.grad.clone()
print("F.conv2d weight temp vs stored: input-grad identical:", torch.equal(g_tmp, g_par), " max|dgrad|:", (g_tmp - g_par).abs().max().item())
# run-to-run determinism of the same conv backward
xi = x.clone().requires_grad_(); F.conv2d(xi, wt, conv_sn.bias, padding=1).backward(g); g_par2 = xi.grad.clone()
print("same call twice: identical:", torch.equal(g_par, g_par2))
# is the div-by-sigma temp a different *layout*? (SN computes weight_orig / sigma; strides)
print("SN weight strides", w_sn.stride(), "baked", conv_sn.weight.stride())

# whole generator: latent grads with SN vs baked, then per-module-type isolation for Linear
from big_sleep.big_sleep import Latents
torch.manual_seed(1)
gan = bg.BigGAN.from_pretrained("biggan-deep-128", freeze=False).to(dev).eval().requires_grad_(False)
lat = Latents(num_latents=len(gan.config.layers) + 1, num_classes=gan.config.num_classes, z_dim=gan.config.z_dim).to(dev)
def grads(model):
    lat.zero_grad(); out = model(*lat(), 1); out.sum().backward()
    return out.detach().clone(), lat.normu.grad.clone(), lat.cls.grad.clone()
o1, gn1, gc1 = grads(gan)
o1b, gn1b, gc1b = grads(gan)
print("SN model run-to-run: out", torch.equal(o1, o1b), "normu grad", torch.equal(gn1, gn1b), "cls grad", torch.equal(gc1, gc1b))
bg.SLICE_CONV_TO_RGB = False  # isolate the spectral-norm bake from the conv_to_rgb slice
gan.freeze_for_inference()
o2, gn2, gc2 = grads(gan)
print("SN vs baked, SN only (128px): out identical", torch.equal(o1, o2), " normu grad identical", torch.equal(gn1, gn2), f"max|d| {(gn1-gn2).abs().max().item():.2e}", " cls grad identical", torch.equal(gc1, gc2), f"max|d| {(gc1-gc2).abs().max().item():.2e}")
lin_sn = bg.snlinear(in_features=256, out_features=2048, eps=1e-12).to(dev).eval().requires_grad_(False)
xl = torch.randn(1, 256, device=dev); gl = torch.randn(1, 2048, device=dev)
xi = xl.clone().requires_grad_(); lin_sn(xi).backward(gl); ga = xi.grad.clone()
nn.utils.remove_spectral_norm(lin_sn); lin_sn.requires_grad_(False)
xi = xl.clone().requires_grad_(); lin_sn(xi).backward(gl); gb = xi.grad.clone()
print("snlinear [1,256]->[1,2048]: input-grad identical:", torch.equal(ga, gb), f"max|d| {(ga-gb).abs().max().item():.2e}")
emb_sn = bg.snlinear(in_features=1000, out_features=128, bias=False, eps=1e-12).to(dev).eval().requires_grad_(False)
xe = torch.rand(15, 1000, device=dev); ge = torch.randn(15, 128, device=dev)
xi = xe.clone().requires_grad_(); emb_sn(xi).backward(ge); ga = xi.grad.clone()
nn.utils.remove_spectral_norm(emb_sn); emb_sn.requires_grad_(False)
xi = xe.clone().requires_grad_(); emb_sn(xi).backward(ge); gb = xi.grad.clone()
print("snlinear [15,1000]->[15,128] (class embedding shape): input-grad identical:", torch.equal(ga, gb), f"max|d| {(ga-gb).abs().max().item():.2e}")
