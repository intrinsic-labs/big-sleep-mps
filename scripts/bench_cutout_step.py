"""Run the canonical step harness with experimental Metal nearest cutouts.

Usage: python scripts/bench_cutout_step.py 512 96 --n 20
Research only: this corrects affected MPS resize gradients and changes seeded
trajectories. The application remains untouched; the patch lives in this process.
The strict source markers make it fail explicitly if the reference loop changes.
"""
import inspect
from pathlib import Path
import runpy
import sys
import textwrap

from bench_cutouts import sample_boxes
from cutout_metal import metal_cutouts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import big_sleep.big_sleep as implementation

    source = textwrap.dedent(inspect.getsource(implementation.BigSleep.forward))
    start = source.index("    pieces = []\n")
    end = source.index("    into = self.normalize_image(into)\n", start)
    replacement = '''    if (self.experimental_resample or self.center_bias
            or self.interpolation_settings["mode"] != "nearest"):
        raise ValueError("This experiment supports only the default nearest cutouts")
    boxes = sample_boxes(width, num_cutouts)
    metadata = torch.tensor(boxes, dtype=torch.int32, device=out.device)
    into = metal_cutouts(out, metadata)

'''
    namespace = dict(vars(implementation), sample_boxes=sample_boxes, metal_cutouts=metal_cutouts)
    exec(compile(source[:start] + replacement + source[end:], __file__, "exec"), namespace)
    implementation.BigSleep.forward = namespace["forward"]
    sys.argv[0] = str(ROOT / "scripts/bench_step.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
