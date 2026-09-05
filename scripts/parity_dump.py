"""Run K seeded Big Sleep steps and dump the EMA image + loss for cross-branch parity checks.

    PYTHONPATH=$PWD python scripts/parity_dump.py 128 8 30 /tmp/parity-base-128.pt
    PYTHONPATH=$PWD python scripts/parity_compare.py /tmp/parity-base-128.pt /tmp/parity-new-128.pt
"""
import sys, torch
size, cutouts, steps, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
flags = set(sys.argv[5:]); sys.argv = sys.argv[:1]
import big_sleep.clip as clip_mod, big_sleep.biggan as biggan_mod
if "--no-patch-embed" in flags: clip_mod.PATCH_EMBED_AS_MATMUL = False
if "--no-fused-bn" in flags: biggan_mod.FUSED_CONDITIONAL_BN = False
if "--no-bake" in flags: biggan_mod.BigGAN.freeze_for_inference = lambda self: self
from big_sleep.big_sleep import Imagine
im = Imagine(text="a pyramid made of ice", image_size=size, num_cutouts=cutouts, epochs=1,
             iterations=steps, save_every=10**9, open_folder=False, seed=1)
losses = []
for i in range(steps):
    _, loss = im.train_step(0, i)
    losses.append(loss.item())
with torch.no_grad():
    im.model.model.latents.eval()
    img = im.model.model()
torch.save({"img": img.cpu(), "losses": torch.tensor(losses),
            "lat": im.model.model.latents.model.normu.detach().cpu(),
            "cls": im.model.model.latents.model.cls.detach().cpu()}, out)
print(f"saved {out}: losses " + " ".join(f"{l:.3f}" for l in losses[:6]) + f" ... {losses[-1]:.3f}")
