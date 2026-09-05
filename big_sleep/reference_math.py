"""The one switch that turns every rounding-changing optimisation in this fork off.

`BIG_SLEEP_REFERENCE_MATH=1` restores the upstream arithmetic everywhere (CLIP patch
embedding as a conv, eager QuickGELU/LayerNorm, unfused conditional BN, the full
128-channel conv_to_rgb, PyTorch cutouts) and reproduces pre-optimisation runs bit for
bit at roughly the pre-optimisation speed. See docs/performance.md, "Parity".
"""
import os

REFERENCE_MATH = os.environ.get('BIG_SLEEP_REFERENCE_MATH', '') not in ('', '0')
