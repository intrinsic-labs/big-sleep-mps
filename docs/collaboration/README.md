# The 2026-09-05 performance pass — primary sources

Verbatim copies of the three files two agents worked from and wrote to while producing the
performance work consolidated in [`docs/performance.md`](../performance.md). They are the primary
source for any write-up of how the pass went; nothing here has been edited.

| file | what it is |
|---|---|
| [`BRIEF.md`](BRIEF.md) | Asher Pope's brief to both agents: the ask ("go super low level… hand-roll anything"), the state of play, the definition of done, the lane split, the rules. |
| [`fable.md`](fable.md) | Claude **Fable 5.1**'s mailbox — its dated entries to the shared `/tmp/big-sleep-perf/` mailbox on branch `perf/fable` (CLIP fwd/bwd + BigGAN lane): lane claim, the stage profile, the patch-embedding find, the Metal kernels, its final summary. |
| [`astra.md`](astra.md) | GPT-6 **Astra**'s (Codex) mailbox on `perf/astra` (cutouts / losses / optimiser / syncs / pipeline lane): baseline measurements, the GPU-lock episode, the normalization cache, the withdrawn latent-moment vectorisation, the MPS nearest-backward bug, the Metal cutout kernel, its final summary. |

How to read them: the two mailboxes interleave by timestamp (both UTC). Each agent read the
other's file before every new line of work, so a claim in one is often answered in the other a
few minutes later — the PYTHONPATH trap at 13:21/13:27, the measurement overlaps that led to
`gpu_run.py` (a `flock` runner) at 13:29/13:38, the lost contiguous hook at 14:10/14:12. Every
number cited in [`docs/performance.md`](../performance.md) traces to a line in one of these
files or to the two branches' notes under [`docs/history/`](../history/).

The two agents' own deliverables — `docs/perf-notes-fable.md` and `docs/perf-notes-astra.md`,
referenced throughout — are preserved under [`docs/history/`](../history/).
