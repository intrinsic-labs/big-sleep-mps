# big-sleep-mps — deep performance pass (Asher Pope, Intrinsic Labs, 2026-09-05)

Asher's ask, in his words: "go super low level and think hard about if it's really as optimized as it
can possibly be. I would like to push past what we can do with public libraries… If we want to hand roll
anything to make it run faster and better on Apple Silicon I'm there for it. Our whole point here is
open source projects, improving that you can do insane low level real work with agents." Maybe there are
no more improvements — that is an acceptable finding — but try with this mindset before we call it good.

## State of play
Repo: big-sleep-mps (MIT fork of lucidrains/big-sleep: CLIP ViT-B/32 + BigGAN-deep-512, gradient
ascent on latents through CLIP). Yesterday's PR (branch `fix/mps-contiguous-grad`, your base) took a
step at CLI defaults from 128 s → 1.02 s on this M1 Max 32 GB / torch 2.14 by (a) a `.contiguous()`
hook on CLIP conv1's output — the trigger is the exact layout `cat().narrow()` produces (permuted +
storage offset + over-allocated batch stride), which sends `mps_convolution_backward` down an ~11 s
path — and (b) freezing CLIP/BigGAN. Read `README.md`, `docs/pytorch-issue-draft.md`,
`scripts/bench_step.py` (the harness; keep using it so numbers stay comparable) and
`scripts/mps_conv_backward_repro.py` before anything else. Audit background:
~/Documents/obsidian/intrinsic-labs-v01/Projects/meta/lab/docs/oss-audit-2026-09-04/big-sleep-mps.md

## What "done" means
1. Profile before you guess: torch.profiler on MPS, `xctrace` Metal System Trace / `MTLCaptureManager`
   if you need GPU-side truth, per-op timings. Find where the 1.02 s actually goes (BigGAN forward?
   CLIP fwd/bwd? the 96 cutouts' interpolate/augment? optimizer/EMA? host↔GPU syncs / `.item()` calls?
   allocator churn? fp32 everywhere?).
2. Then attack it, lowest level first where it pays: fp16/bf16 on MPS where it holds quality, memory
   format, fused ops, killing syncs, `torch.compile` on MPS if it works, MPSGraph, and — explicitly
   welcome — hand-rolled Metal kernels via a PyTorch C++/ObjC++ extension for any op that is provably
   the bottleneck (interpolate/cutouts, the conv path, softmax/attention, upsampling). Own-code beats
   library-code when it measures faster.
3. Quality parity is part of the bar: same seed → same or perceptually equivalent image (document any
   drift, e.g. from fp16). Never trade correctness silently.
4. Negative results count: what you tried that did NOT help, with numbers, goes in the notes.
5. Measure everything with `scripts/bench_step.py` (synchronised means, ≥ 20 steps after warmup) at
   CLI defaults (512 px / 96 cutouts), `--fast`, and 128 px / 8 cutouts. End-to-end wall time for
   `dream "a pyramid made of ice" --fast` too. Keep individual runs to minutes; leave nothing running.

## Working together
Two of you are on this in parallel: Fable 5.1 (Claude Code) on branch `perf/fable`, worktree
~/.cache/intrinsic-vault/worktrees/big-sleep-fable; GPT-6 Astra (Codex) on branch `perf/astra`,
worktree ~/.cache/intrinsic-vault/worktrees/big-sleep-astra. Shared mailbox: /tmp/big-sleep-perf/ —
append dated entries to YOUR file (fable.md / astra.md); READ the other's file before each new line of
work and claim a lane there before digging in so you split rather than duplicate. Suggested opening
split (renegotiate in the mailbox if the profile says otherwise): Fable → CLIP fwd/bwd + BigGAN
forward (dtype, layout, fusion, compile, custom kernels for the conv/attention path); Astra → the
cutout/augmentation pipeline, optimizer/EMA/latent update, sync elimination, allocator/memory, and
end-to-end pipeline structure, plus custom Metal where a torch op there is the bottleneck. If one of
you finds the other's lane is where the time is, say so in the mailbox and re-split. Cite each other's
numbers rather than re-measuring.

## Rules
Own branch only. Commit small, one idea each, message ends with the co-author trailer below. Never
touch `main` or `fix/mps-contiguous-grad`, never merge, never publish, never change GitHub settings.
Deliver `docs/perf-notes-<fable|astra>.md`: what the profile said, the ranked table of what you changed
with before/after numbers, what failed and why, and what you believe the remaining floor is and why.
Final mailbox entry summarising the same. Git trailer for commits:
Co-Authored-By: <your model name> <noreply@anthropic.com or noreply@openai.com>
Claude-Session: https://claude.ai/code/session_01PwznaY5HxQjyzCT7ABn4PE
