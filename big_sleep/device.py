"""One place that decides which device Big Sleep runs on.

Order: MPS (Apple Silicon) -> CUDA -> CPU, or whatever BIG_SLEEP_DEVICE names
(e.g. `BIG_SLEEP_DEVICE=cpu` to exercise the CPU path on a Mac). Every module
imports DEVICE from here instead of re-deriving it from `platform.processor()`,
which reports 'i386' under a Rosetta Python and silently sent Macs to the CPU.
"""
import os

import torch


def pick_device() -> torch.device:
    forced = os.environ.get('BIG_SLEEP_DEVICE')
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device='mps')  # a tiny op proves the backend actually works
            return torch.device('mps')
        except Exception as e:  # pragma: no cover - depends on the host
            print(f'⚠️ MPS is available but failed a test op ({e}); falling back to CPU')
            return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = pick_device()
IS_MPS = DEVICE.type == 'mps'


def synchronize():
    """Block until queued work on DEVICE has finished (for timing)."""
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def describe() -> str:
    if DEVICE.type == 'mps':
        return 'Apple MPS (Metal Performance Shaders)'
    if DEVICE.type == 'cuda':
        return f'CUDA ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})'
    return 'CPU (this will be slow)'
