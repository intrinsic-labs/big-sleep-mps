"""Per-component fwd/bwd timings of CLIP ViT-B/32 on MPS at Big Sleep's cutout batch.

    PYTHONPATH=$PWD python scripts/clip_microbench.py 96

Isolates: whole encode_image fwd / bwd (input grad only, weights frozen) in fp32 / fp16 / bf16;
one ResidualAttentionBlock split into nn.MultiheadAttention vs an explicit SDPA rewrite,
MLP, LayerNorm; and a torch.compile probe.
"""
import sys, time, copy
import torch, torch.nn as nn, torch.nn.functional as F

N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
dev = "mps"


def sync(): torch.mps.synchronize()


def bench(label, fn, n=5, warm=2):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); ms = (time.perf_counter() - t) / n * 1000
    print(f"{label:60s} {ms:9.1f} ms"); return ms


def fwd_bwd(model, x, dtype):
    def run():
        xi = x.to(dtype).detach().requires_grad_()
        y = model(xi)
        y.float().sum().backward()
    return run


def fwd_only(model, x, dtype):
    def run():
        with torch.no_grad():
            model(x.to(dtype))
    return run


from big_sleep.clip import load


def fresh(dtype):
    """As shipped, CLIP on MPS is fp16 for Linear/Conv/proj with fp32 LayerNorms
    (clip.convert_weights). deepcopy of an MPS module aliases storage, so load fresh."""
    perceptor, _ = load("ViT-B/32", device=dev, jit=False)
    perceptor.requires_grad_(False)
    v = perceptor.visual
    if dtype == torch.float32:
        v.float()
    elif dtype == torch.bfloat16:
        for mod in v.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.data = mod.weight.data.bfloat16()
                if mod.bias is not None: mod.bias.data = mod.bias.data.bfloat16()
            if isinstance(mod, nn.MultiheadAttention):
                mod.in_proj_weight.data = mod.in_proj_weight.data.bfloat16()
                mod.in_proj_bias.data = mod.in_proj_bias.data.bfloat16()
        v.proj.data = v.proj.data.bfloat16()
    return v


x = torch.randn(N, 3, 224, 224, device=dev)
print(f"== CLIP ViT-B/32 encode_image, batch {N}, torch {torch.__version__} ==")
for dtype, label in ((torch.float16, "fp16 (as shipped, fp32 LN)"), (torch.float32, "fp32 (model.float())"), (torch.bfloat16, "bf16 weights, fp32 LN")):
    m = fresh(dtype)
    print("  conv1 dtype", m.conv1.weight.dtype, "ln_pre dtype", m.ln_pre.weight.dtype)
    bench(f"encode_image fwd (no_grad)  {label}", fwd_only(m, x, dtype))
    bench(f"encode_image fwd+bwd(input) {label}", fwd_bwd(m, x, dtype))
    del m

visual = fresh(torch.float32)

# ---- one block, fp32, LND layout as in CLIP ----
blk = visual.transformer.resblocks[0]
L, D = 50, 768
h = torch.randn(L, N, D, device=dev)
print(f"\n== one ResidualAttentionBlock, x:[{L},{N},{D}] fp32 ==")
bench("block fwd+bwd", fwd_bwd(blk, h, torch.float32))
bench("  nn.MultiheadAttention fwd+bwd", fwd_bwd(lambda t: blk.attn(t, t, t, need_weights=False)[0], h, torch.float32))
bench("  ln_1 fwd+bwd", fwd_bwd(blk.ln_1, h, torch.float32))
bench("  mlp fwd+bwd", fwd_bwd(blk.mlp, h, torch.float32))
bench("  c_fc linear only fwd+bwd", fwd_bwd(blk.mlp.c_fc, h, torch.float32))
bench("  QuickGELU only fwd+bwd", fwd_bwd(blk.mlp.gelu, torch.randn(L, N, 4 * D, device=dev), torch.float32))

# explicit attention, batch-first, SDPA
W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
Wo, bo = blk.attn.out_proj.weight, blk.attn.out_proj.bias
nh = blk.attn.num_heads


def attn_sdpa(t):  # t: [N, L, D]
    qkv = F.linear(t, W, b).reshape(N, L, 3, nh, D // nh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    o = F.scaled_dot_product_attention(q, k, v)
    return F.linear(o.transpose(1, 2).reshape(N, L, D), Wo, bo)


def attn_manual(t):
    qkv = F.linear(t, W, b).reshape(N, L, 3, nh, D // nh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    a = (q @ k.transpose(-1, -2) * (D // nh) ** -0.5).softmax(-1)
    o = a @ v
    return F.linear(o.transpose(1, 2).reshape(N, L, D), Wo, bo)


hb = h.transpose(0, 1).contiguous()
bench("  explicit SDPA attention (batch-first) fwd+bwd", fwd_bwd(attn_sdpa, hb, torch.float32))
bench("  explicit manual softmax attention (batch-first) fwd+bwd", fwd_bwd(attn_manual, hb, torch.float32))
with torch.no_grad():
    ref = blk.attn(h, h, h, need_weights=False)[0].transpose(0, 1)
    print("  max|sdpa - mha| =", (attn_sdpa(hb) - ref).abs().max().item())

print("\n== fp16 same block (as shipped: fp16 linears, fp32 LN) ==")
blk16 = fresh(torch.float16).transformer.resblocks[0]; h16 = h.half()
bench("block fwd+bwd fp16", fwd_bwd(blk16, h16, torch.float16))
bench("  nn.MultiheadAttention fwd+bwd fp16", fwd_bwd(lambda t: blk16.attn(t, t, t, need_weights=False)[0], h16, torch.float16))
bench("  mlp fwd+bwd fp16", fwd_bwd(blk16.mlp, h16, torch.float16))

if "--compile" in sys.argv:
    print("\n== torch.compile probe ==")
    try:
        cm = torch.compile(copy.deepcopy(visual))
        bench("compiled encode_image fwd+bwd fp32", fwd_bwd(cm, x, torch.float32), n=3, warm=1)
    except Exception as e:
        print("torch.compile failed:", type(e).__name__, str(e)[:300])
