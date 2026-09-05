import fire
import random as rnd
import os
import platform
import torch
from big_sleep import Imagine
from big_sleep.device import DEVICE, describe
from .version import __version__

# lucidrains' schedule was tuned for a V100 doing ~4 it/s. On an M1 Max a
# 512 px / 96-cutout step is ~1 s, so the defaults below finish in minutes;
# --upstream_defaults restores the originals.
UPSTREAM_EPOCHS, UPSTREAM_ITERATIONS, UPSTREAM_CUTOUTS = 20, 1050, 128

def check_environment():
    """Print which device big-sleep will run on."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║                Big Sleep Environment                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    icon = "✅" if DEVICE.type in ("mps", "cuda") else "⚠️ "
    print(f"{icon} Device: {describe()}")
    if DEVICE.type == "cpu" and platform.system() == "Darwin" and platform.machine() == "arm64":
        print("   This is an Apple Silicon Mac but MPS is unavailable — check that this")
        print("   Python is arm64 (not Rosetta) and that torch>=2.0 is installed.")
    print(f"• PyTorch version: {torch.__version__}")
    print(f"• Python version: {platform.python_version()}")
    print("────────────────────────────────────────────────────────")


def train(
    text=None,
    img=None,
    text_min="",
    lr = .07,
    image_size = 512,
    gradient_accumulate_every = 1,
    epochs = 1,
    iterations = 500,
    save_every = 50,
    overwrite = False,
    save_progress = False,
    save_date_time = False,
    bilinear = False,
    open_folder = True,
    seed = 0,
    append_seed = False,
    random = False,
    torch_deterministic = False,
    max_classes = None,
    class_temperature = 2.,
    save_best = True,  # Changed to True to save best result by default
    experimental_resample = False,
    ema_decay = 0.5,
    num_cutouts = 96,
    center_bias = False,  # Matching original default
    larger_model = False,
    output_dir = None,
    fast = False,  # 1 x 200 steps, 64 cutouts: a preview in a couple of minutes
    upstream_defaults = False,  # lucidrains' 20 x 1050 x 128-cutout schedule
    debug = False
):
    print(f'Starting up... v{__version__}')

    if random:
        seed = rnd.randint(0, 1e6)
        
    # Handle output directory
    if output_dir:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Images will be saved to: {os.path.abspath(output_dir)}")

    if fast and upstream_defaults:
        raise SystemExit("--fast and --upstream_defaults are mutually exclusive")
    if fast:
        epochs, iterations, num_cutouts = 1, 200, 64
        print("⚡ Fast mode - 1 x 200 iterations, 64 cutouts")
    if upstream_defaults:
        epochs, iterations, num_cutouts = UPSTREAM_EPOCHS, UPSTREAM_ITERATIONS, UPSTREAM_CUTOUTS
        print(f"Using upstream schedule - {epochs} x {iterations} iterations, {num_cutouts} cutouts")
        
    # Set the debug flag in the big_sleep module
    import big_sleep.big_sleep
    big_sleep.big_sleep.DEBUG = debug
    
    if debug:
        print("🔍 Debug mode enabled - verbose output will be shown")
        
    imagine = Imagine(
        text=text,
        img=img,
        text_min=text_min,
        lr = lr,
        image_size = image_size,
        gradient_accumulate_every = gradient_accumulate_every,
        epochs = epochs,
        iterations = iterations,
        save_every = save_every,
        save_progress = save_progress,
        bilinear = bilinear,
        seed = seed,
        append_seed = append_seed,
        torch_deterministic = torch_deterministic,
        open_folder = open_folder,
        max_classes = max_classes,
        class_temperature = class_temperature,
        save_date_time = save_date_time,
        save_best = save_best,
        experimental_resample = experimental_resample,
        ema_decay = ema_decay,
        num_cutouts = num_cutouts,
        center_bias = center_bias,
        larger_clip = larger_model,
        output_dir = output_dir
    )

    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            exit()

    imagine()
    
    # Print completion message
    print("\n╔════════════════════════════════════════════════════╗")
    print("║               Generation Complete!                  ║") 
    print("╚════════════════════════════════════════════════════╝")
    abs_path = os.path.abspath(str(imagine.filename))
    print(f"Image saved to: {abs_path}")
    print("────────────────────────────────────────────────────────")


def main():
    check_environment()
    fire.Fire(train)
