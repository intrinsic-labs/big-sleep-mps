import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
d = (a["img"] - b["img"]).abs()
mse = (d ** 2).mean().item()
psnr = 10 * torch.log10(torch.tensor(1.0) / mse).item() if mse > 0 else float("inf")
print(f"image  max|diff| {d.max().item():.4f}  mean|diff| {d.mean().item():.5f}  PSNR {psnr:.1f} dB  (8-bit quantum = 0.0039)")
print(f"px >1 quantum off: {(d > 1/255).float().mean().item()*100:.2f}%")
print(f"losses max|diff| {(a['losses'] - b['losses']).abs().max().item():.4f}  final {a['losses'][-1]:.3f} vs {b['losses'][-1]:.3f}")
print(f"latents max|diff| {(a['lat'] - b['lat']).abs().max().item():.2e}  cls max|diff| {(a['cls'] - b['cls']).abs().max().item():.2e}")
