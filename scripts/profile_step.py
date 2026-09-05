"""Where does one Big Sleep step go on MPS?  Stage timings + torch.profiler op table.

Usage:
    python scripts/profile_step.py 512 96        # CLI defaults
    python scripts/profile_step.py 128 8 --ops   # also print the per-op CPU table

Stage timings synchronise the device between stages, so they are GPU-inclusive
wall times (MPS runs async; unsynchronised per-op CPU times only measure
kernel *encoding*). `--ops` adds torch.profiler's aggregated table, which on MPS
is still useful for op *counts* and for spotting host<->device copies/syncs.
"""
import argparse, sys, time
import torch
import torch.nn.functional as F


def sync():
    torch.mps.synchronize()


def timed(label, fn, acc, n=1):
    sync(); t = time.perf_counter()
    for _ in range(n):
        r = fn()
    sync(); dt = (time.perf_counter() - t) / n
    acc.append((label, dt * 1000))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_size", type=int)
    ap.add_argument("num_cutouts", type=int)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ops", action="store_true")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]

    from big_sleep.big_sleep import Imagine, rand_cutout
    im = Imagine(text="a pyramid made of ice", image_size=args.image_size,
                 num_cutouts=args.num_cutouts, epochs=1, iterations=1,
                 save_every=10**9, open_folder=False, seed=1)
    bs = im.model
    for _ in range(2):
        im.train_step(0, 0)
    sync()

    # ---- staged step (same math as BigSleep.forward + train_step) ----
    rows = []
    for _ in range(args.n):
        acc = []
        out = timed("biggan fwd", lambda: bs.model(), acc)
        def cutouts():
            pieces = []
            for _ in range(bs.num_cutouts):
                size = int(bs.image_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
                pieces.append(F.interpolate(rand_cutout(out, size), (224, 224), **bs.interpolation_settings))
            return bs.normalize_image(torch.cat(pieces))
        into = timed("cutouts+cat+normalize", cutouts, acc)
        img_embed = timed("clip encode_image fwd", lambda: bs.perceptor.encode_image(into), acc)
        def losses():
            lat, cls = bs.model.latents()
            lat_loss = torch.abs(1 - torch.std(lat, dim=1)).mean() + torch.abs(torch.mean(lat, dim=1)).mean() + 4 * torch.max(torch.square(lat).mean(), bs.model.latents.model.thresh_lat)
            for a in lat:
                m = a.mean(); d = a - m; v = (d ** 2).mean(); s = v ** .5; z = d / s
                lat_loss = lat_loss + torch.abs((z ** 4).mean() - 3) / lat.shape[0] + torch.abs((z ** 3).mean()) / lat.shape[0]
            cls_loss = ((50 * torch.topk(cls, largest=False, dim=1, k=999)[0]) ** 2).mean()
            sim = sum(bs.sim_txt_to_img(t, img_embed) for t in im.encoded_texts["max"]).mean()
            return lat_loss + cls_loss + sim
        loss = timed("losses", losses, acc)
        timed("backward (clip+cutouts+biggan)", lambda: loss.backward(), acc)
        timed("adam+ema+zero_grad", lambda: (im.optimizer.step(), bs.model.latents.update(), im.optimizer.zero_grad()), acc)
        rows.append(acc)
    print(f"\n== stage means over {args.n} steps, {args.image_size}px / {args.num_cutouts} cutouts, torch {torch.__version__} ==")
    tot = 0
    for i, (label, _) in enumerate(rows[0]):
        m = sum(r[i][1] for r in rows) / len(rows); tot += m
        print(f"  {label:34s} {m:8.1f} ms")
    print(f"  {'TOTAL':34s} {tot:8.1f} ms")

    # backward split: clip-only backward vs full
    acc = []
    out = bs.model().detach().requires_grad_()
    pieces = [F.interpolate(rand_cutout(out, int(bs.image_size * .8)), (224, 224), **bs.interpolation_settings) for _ in range(bs.num_cutouts)]
    into = bs.normalize_image(torch.cat(pieces)).detach().requires_grad_()
    e = bs.perceptor.encode_image(into); l = e.sum()
    timed("  clip bwd only (to cutout batch)", lambda: l.backward(), acc)
    pieces = [F.interpolate(rand_cutout(out, int(bs.image_size * .8)), (224, 224), **bs.interpolation_settings) for _ in range(bs.num_cutouts)]
    into = bs.normalize_image(torch.cat(pieces))
    timed("  cutouts bwd only (to biggan out)", lambda: into.backward(torch.randn_like(into)), acc)
    lat, cls = bs.model.latents()
    o = bs.model.biggan(lat, cls, 1)
    timed("  biggan bwd only (to latents)", lambda: o.backward(torch.randn_like(o)), acc)
    for label, ms in acc:
        print(f"  {label:34s} {ms:8.1f} ms")

    if args.ops:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            im.train_step(0, 0); sync()
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))


if __name__ == "__main__":
    main()
